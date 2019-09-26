# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
import sys
from functools import partial

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import numpy as np


import tensorflow as tf

from prepro_utils import preprocess_text, encode_ids
import sentencepiece as spm


special_symbols = {
    "<unk>"  : 0,
    "<s>"    : 1,
    "</s>"   : 2,
    "<pad>"  : 3,
    "<eod>"  : 4,
    "<eop>"  : 5,
    "<hi>"   : 6,
    "<eng>"   : 7
}

VOCAB_SIZE = 32000
UNK_ID = special_symbols["<unk>"]
EOD_ID = special_symbols["<eod>"]
EOP_ID = special_symbols["<eop>"]
HIN_ID = special_symbols["<hi>"]
ENG_ID = special_symbols["<eng>"]

def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))



def format_filename_gen(prefix, bsz_per_host, seq_len, bi_data, suffix,
                    uncased=False):
  """docs."""
  if not uncased:
    uncased_str = ""
  else:
    uncased_str = "uncased."
  if bi_data:
    bi_data_str = "bi"
  else:
    bi_data_str = "uni"


  file_name = "{}.bsz-{}.seqlen-{}.{}{}.gen.{}".format(
      prefix, bsz_per_host, seq_len, uncased_str, 
      bi_data_str, suffix)

  return file_name

def _create_data(idx, input_paths, transliterate=True, language_tag=True, major_language='english'):
  # Load sentence-piece model
  sp = spm.SentencePieceProcessor()
  sp.Load(FLAGS.sp_path)

  input_shards = []
  total_line_cnt = 0
  for input_path in input_paths:
    input_data, sent_ids = [], []
    sent_id, line_cnt = True, 0
    tf.logging.info("Processing %s", input_path)
    for line in tf.gfile.Open(input_path):
      if line_cnt % 100000 == 0:
        tf.logging.info("Loading line %d", line_cnt)
      line_cnt += 1

      if not line.strip():
        if FLAGS.use_eod:
          sent_id = not sent_id
          cur_sent = [EOD_ID]
        else:
          continue
      else:
        if FLAGS.from_raw_text:
          cur_sent = preprocess_text(line.strip(), lower=FLAGS.uncased)
          cur_sent = encode_ids(sp, cur_sent,
                               transliterate=transliterate, language_tag=language_tag,
                               eng_id=ENG_ID, hin_id=HIN_ID, major_language=major_language)
        else:
          cur_sent = list(map(int, line.strip().split()))
        if FLAGS.use_eop:
          cur_sent.append(EOP_ID)

      input_data.extend(cur_sent)
      sent_ids.extend([sent_id] * len(cur_sent))
      sent_id = not sent_id

    tf.logging.info("Finish with line %d", line_cnt)
    if line_cnt == 0:
      continue

    input_data = np.array(input_data, dtype=np.int64)
    sent_ids = np.array(sent_ids, dtype=np.bool)

    total_line_cnt += line_cnt
    input_shards.append((input_data, sent_ids))

  tf.logging.info("[Task %d] Total number line: %d", idx, total_line_cnt)

  tfrecord_dir = os.path.join(FLAGS.save_dir, "tfrecords")

  filenames, num_batch = [], 0

  # Randomly shuffle input shards (with a fixed but distinct random seed)
  np.random.seed(100 * FLAGS.task + FLAGS.pass_id)

  perm_indices = np.random.permutation(len(input_shards))
  tf.logging.info("Using perm indices %s for pass %d",
                  perm_indices.tolist(), FLAGS.pass_id)
  

  input_data_list, sent_ids_list = [], []
  prev_sent_id = None
  for perm_idx in perm_indices:
    input_data, sent_ids = input_shards[perm_idx]
    # make sure the `send_ids[0] == not prev_sent_id`
    if prev_sent_id is not None and sent_ids[0] == prev_sent_id:
      sent_ids = np.logical_not(sent_ids)

    # append to temporary list
    input_data_list.append(input_data)
    sent_ids_list.append(sent_ids)

    # update `prev_sent_id`
    prev_sent_id = sent_ids[-1]

  input_data = np.concatenate(input_data_list)
  sent_ids = np.concatenate(sent_ids_list)

  file_name, cur_num_batch = create_tfrecords(
      save_dir=tfrecord_dir,
      basename="{}-{}-{}".format(FLAGS.split, idx, FLAGS.pass_id),
      data=[input_data, sent_ids],
      bsz_per_host=FLAGS.bsz_per_host,
      seq_len=FLAGS.seq_len,
      bi_data=FLAGS.bi_data,
      sp=sp,
  )

  filenames.append(file_name)
  num_batch += cur_num_batch

  record_info = {
      "filenames": filenames,
      "num_batch": num_batch
  }

  return record_info


def create_data(_):
  # Validate FLAGS
  assert FLAGS.bsz_per_host % FLAGS.num_core_per_host == 0
  if not FLAGS.use_tpu:
    FLAGS.num_core_per_host = 1  # forced to be one

  # Make workdirs
  if not tf.gfile.Exists(FLAGS.save_dir):
    tf.gfile.MakeDirs(FLAGS.save_dir)

  tfrecord_dir = os.path.join(FLAGS.save_dir, "tfrecords")
  if not tf.gfile.Exists(tfrecord_dir):
    tf.gfile.MakeDirs(tfrecord_dir)

  # Create and dump corpus_info from task 0
  if FLAGS.task == 0:
    corpus_info = {
        "vocab_size": VOCAB_SIZE,
        "bsz_per_host": FLAGS.bsz_per_host,
        "num_core_per_host": FLAGS.num_core_per_host,
        "seq_len": FLAGS.seq_len,
        "uncased": FLAGS.uncased,
        "bi_data": FLAGS.bi_data,
        "use_eod": FLAGS.use_eod,
        "use_eop": FLAGS.use_eop,
        "sp_path": FLAGS.sp_path,
        "input_glob": FLAGS.input_glob,
    }
    corpus_info_path = os.path.join(FLAGS.save_dir, "corpus_info.json")
    with tf.gfile.Open(corpus_info_path, "w") as fp:
      json.dump(corpus_info, fp)

  # Interleavely split the work into FLAGS.num_task splits
  file_paths = sorted(tf.gfile.Glob(FLAGS.input_glob))
  tf.logging.info("Use glob: %s", FLAGS.input_glob)
  tf.logging.info("Find %d files: %s", len(file_paths), file_paths)

  task_file_paths = file_paths[FLAGS.task::FLAGS.num_task]
  if not task_file_paths:
    tf.logging.info("Exit: task %d has no file to process.", FLAGS.task)
    return

  tf.logging.info("Task %d process %d files: %s",
                  FLAGS.task, len(task_file_paths), task_file_paths)

  if FLAGS.bi_data:
    tf.logging.info("Using bi data")

  record_info = _create_data(FLAGS.task, task_file_paths, 
                             transliterate=FLAGS.transliterate, 
                             language_tag=FLAGS.language_tag,
                             major_language=FLAGS.major_language)

  record_prefix = "record_info-{}-{}-{}".format(
      FLAGS.split, FLAGS.task, FLAGS.pass_id)
  record_name = format_filename_gen(
      prefix=record_prefix,
      bsz_per_host=FLAGS.bsz_per_host,
      seq_len=FLAGS.seq_len,
      bi_data=FLAGS.bi_data,
      suffix="json",
      uncased=FLAGS.uncased)
  record_info_path = os.path.join(tfrecord_dir, record_name)

  with tf.gfile.Open(record_info_path, "w") as fp:
    json.dump(record_info, fp)


def batchify(data, bsz_per_host, sent_ids=None):
  num_step = len(data) // bsz_per_host
  data = data[:bsz_per_host * num_step]
  data = data.reshape(bsz_per_host, num_step)
  if sent_ids is not None:
    sent_ids = sent_ids[:bsz_per_host * num_step]
    sent_ids = sent_ids.reshape(bsz_per_host, num_step)

  if sent_ids is not None:
    return data, sent_ids
  return data




def create_tfrecords(save_dir, basename, data, bsz_per_host, seq_len,
                     bi_data, sp):
  data, sent_ids = data[0], data[1]

  num_core = FLAGS.num_core_per_host
  bsz_per_core = bsz_per_host // num_core

  if bi_data:
    assert bsz_per_host % (2 * FLAGS.num_core_per_host) == 0
    fwd_data, fwd_sent_ids = batchify(data, bsz_per_host // 2, sent_ids)

    fwd_data = fwd_data.reshape(num_core, 1, bsz_per_core // 2, -1)
    fwd_sent_ids = fwd_sent_ids.reshape(num_core, 1, bsz_per_core // 2, -1)

    bwd_data = fwd_data[:, :, :, ::-1]
    bwd_sent_ids = fwd_sent_ids[:, :, :, ::-1]

    data = np.concatenate(
        [fwd_data, bwd_data], 1).reshape(bsz_per_host, -1)
    sent_ids = np.concatenate(
        [fwd_sent_ids, bwd_sent_ids], 1).reshape(bsz_per_host, -1)
  else:
    data, sent_ids = batchify(data, bsz_per_host, sent_ids)

  tf.logging.info("Raw data shape %s.", data.shape)

  file_name = format_filename_gen(
      prefix=basename,
      bsz_per_host=bsz_per_host,
      seq_len=seq_len,
      bi_data=bi_data,
      suffix="tfrecords",
      uncased=FLAGS.uncased)

  save_path = os.path.join(save_dir, file_name)
  record_writer = tf.python_io.TFRecordWriter(save_path)
  tf.logging.info("Start writing %s.", save_path)

  num_batch = 0

  data_len = data.shape[1]
  i = 0
  while i + seq_len+1 <= data_len:
    if num_batch % 500 == 0:
      tf.logging.info("Processing batch %d", num_batch)

    all_ok = True
    features = []
    for idx in range(bsz_per_host):
      cat_data = data[idx,i: i+seq_len]
      tgt = data[idx,i+1: i+seq_len+1]
      is_masked = np.ones(seq_len,dtype=np.int64)
      label = 0
      seg_id = [0]*seq_len

      feature = {
          "input": _int64_feature(cat_data),
          "labels": _int64_feature(tgt),
      }
      features.append(feature)

    if all_ok:
      assert len(features) == bsz_per_host
      for feature in features:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())
      num_batch += 1
    else:
      break

    i += seq_len

  record_writer.close()
  tf.logging.info("Done writing %s. Num of batches: %d", save_path, num_batch)

  return save_path, num_batch


################
# get_input_fn #
################
def _convert_example(example, use_bfloat16):
  """Cast int64 into int32 and float32 to bfloat16 if use_bfloat16."""
  for key in list(example.keys()):
    val = example[key]
    if tf.keras.backend.is_sparse(val):
      val = tf.sparse.to_dense(val)
    if val.dtype == tf.int64:
      val = tf.cast(val, tf.int32)
    if use_bfloat16 and val.dtype == tf.float32:
      val = tf.cast(val, tf.bfloat16)

    example[key] = val


def parse_files_to_dataset(parser, file_names, split, num_batch, num_hosts,
                           host_id, num_core_per_host, bsz_per_core, 
                           toeval=False):
  # list of file pathes
  num_files = len(file_names)
  num_files_per_host = num_files // num_hosts
  my_start_file_id = host_id * num_files_per_host
  my_end_file_id = (host_id + 1) * num_files_per_host
  if host_id == num_hosts - 1:
    my_end_file_id = num_files
  file_paths = file_names[my_start_file_id: my_end_file_id]
  tf.logging.info("Host %d handles %d files", host_id, len(file_paths))

  #assert split == "train"
  dataset = tf.data.Dataset.from_tensor_slices(file_paths)

  # file-level shuffle
  if len(file_paths) > 1 and not toeval:
    dataset = dataset.shuffle(len(file_paths))

  # Note: we cannot perform sample-level shuffle here because this will violate
  # the consecutive requirement of data stream.
  dataset = tf.data.TFRecordDataset(dataset)

  # (zihang): since we are doing online preprocessing, the parsed result of
  # the same input at each time will be different. Thus, cache processed data
  # is not helpful. It will use a lot of memory and lead to contrainer OOM.
  # So, change to cache non-parsed raw data instead.
  
  if not toeval:
    dataset = dataset.cache().map(parser)
    dataset = dataset.repeat()
  else:
    dataset = dataset.map(parser)

  dataset = dataset.batch(bsz_per_core, drop_remainder=True)
  dataset = dataset.prefetch(num_core_per_host * bsz_per_core)

  return dataset

def get_dataset(params, num_hosts, num_core_per_host, split, file_names,
                num_batch, seq_len, use_bfloat16=False, toeval=True):

  bsz_per_core = params["batch_size"]
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0

  #### Function used to parse tfrecord
  def parser(record):
    """function used to parse tfrecord."""
    record_spec = {
        "input": tf.FixedLenFeature([seq_len], tf.int64),
        "labels": tf.FixedLenFeature([seq_len], tf.int64),
    }

    # retrieve serialized example
    example = tf.parse_single_example(
        serialized=record,
        features=record_spec)

    _convert_example(example, use_bfloat16)

    for k, v in example.items():
      tf.logging.info("%s: %s", k, v)

    return example

  # Get dataset
  dataset = parse_files_to_dataset(
      parser=parser,
      file_names=file_names,
      split=split,
      num_batch=num_batch,
      num_hosts=num_hosts,
      host_id=host_id,
      num_core_per_host=num_core_per_host,
      bsz_per_core=bsz_per_core,
      toeval=toeval)

  return dataset


def get_input_fn(
    tfrecord_dir,
    split,
    bsz_per_host,
    seq_len,
    bi_data,
    num_hosts=1,
    num_core_per_host=1,
    uncased=False,
    num_passes=None,
    use_bfloat16=False,
    toeval=False):

  # Merge all record infos into a single one
  record_glob_base = format_filename_gen(
      prefix="record_info-{}-*".format(split),
      bsz_per_host=bsz_per_host,
      seq_len=seq_len,
      bi_data=bi_data,
      suffix="json",
      uncased=uncased)

  record_info = {"num_batch": 0, "filenames": []}

  tfrecord_dirs = tfrecord_dir.split(",")
  tf.logging.info("Use the following tfrecord dirs: %s", tfrecord_dirs)

  for idx, record_dir in enumerate(tfrecord_dirs):
    record_glob = os.path.join(record_dir, record_glob_base)
    tf.logging.info("[%d] Record glob: %s", idx, record_glob)

    record_paths = sorted(tf.gfile.Glob(record_glob))
    tf.logging.info("[%d] Num of record info path: %d",
                    idx, len(record_paths))

    cur_record_info = {"num_batch": 0, "filenames": []}

    for record_info_path in record_paths:
      if num_passes is not None:
        record_info_name = os.path.basename(record_info_path)
        fields = record_info_name.split(".")[0].split("-")
        pass_id = int(fields[-1])
        if len(fields) == 5 and pass_id >= num_passes:
          tf.logging.info("Skip pass %d: %s", pass_id, record_info_name)
          continue

      with tf.gfile.Open(record_info_path, "r") as fp:
        info = json.load(fp)
        if num_passes is not None:
          eff_num_passes = min(num_passes, len(info["filenames"]))
          ratio = eff_num_passes / len(info["filenames"])
          cur_record_info["num_batch"] += int(info["num_batch"] * ratio)
          cur_record_info["filenames"] += info["filenames"][:eff_num_passes]
        else:
          cur_record_info["num_batch"] += info["num_batch"]
          cur_record_info["filenames"] += info["filenames"]

    # overwrite directory for `cur_record_info`
    new_filenames = []
    for filename in cur_record_info["filenames"]:
      basename = os.path.basename(filename)
      new_filename = os.path.join(record_dir, basename)
      new_filenames.append(new_filename)
    cur_record_info["filenames"] = new_filenames

    tf.logging.info("[Dir %d] Number of chosen batches: %s",
                    idx, cur_record_info["num_batch"])
    tf.logging.info("[Dir %d] Number of chosen files: %s",
                    idx, len(cur_record_info["filenames"]))
    tf.logging.info(cur_record_info["filenames"])

    # add `cur_record_info` to global `record_info`
    record_info["num_batch"] += cur_record_info["num_batch"]
    record_info["filenames"] += cur_record_info["filenames"]

  tf.logging.info("Total number of batches: %d",
                  record_info["num_batch"])
  tf.logging.info("Total number of files: %d",
                  len(record_info["filenames"]))
  tf.logging.info(record_info["filenames"])

  def input_fn(params):
    """docs."""
    assert params["batch_size"] * num_core_per_host == bsz_per_host,\
          f'{(params["batch_size"] , num_core_per_host , bsz_per_host)}'

    dataset = get_dataset(
        params=params,
        num_hosts=num_hosts,
        num_core_per_host=num_core_per_host,
        split=split,
        file_names=record_info["filenames"],
        num_batch=record_info["num_batch"],
        seq_len=seq_len,
        use_bfloat16=use_bfloat16,
        toeval=toeval)

    return dataset

  return input_fn, record_info


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  flags.DEFINE_bool("use_tpu", True, help="whether to use TPUs")
  flags.DEFINE_integer("bsz_per_host", 32, help="batch size per host.")
  flags.DEFINE_integer("num_core_per_host", 8, help="num TPU cores per host.")

  flags.DEFINE_integer("seq_len", 512,
                       help="Sequence length.")
  flags.DEFINE_bool("uncased", False, help="Use uncased inputs or not.")
  flags.DEFINE_bool("bi_data", True,
                    help="whether to create bidirectional data")
  flags.DEFINE_bool("use_eod", True,
                    help="whether to append EOD at the end of a doc.")
  flags.DEFINE_bool("use_eop", True,
                    help="whether to append EOP afte each para.")
  flags.DEFINE_bool("from_raw_text", True,
                    help="Whether the input is raw text or encoded ids.")
  flags.DEFINE_string("input_glob", "data/example/*.txt",
                      help="Input file glob.")
  flags.DEFINE_string("sp_path", "", help="Path to the sentence piece model.")
  flags.DEFINE_string("save_dir", "proc_data/example",
                      help="Directory for saving the processed data.")
  flags.DEFINE_enum("split", "train", ["train", "dev", "test"],
                    help="Save the data as which split.")
  flags.DEFINE_integer("pass_id", 0, help="ID of the current pass."
                       "Different passes sample different negative segment.")
  flags.DEFINE_integer("num_task", 1, help="Number of total tasks.")
  flags.DEFINE_integer("task", 0, help="The Task ID. This value is used when "
                       "using multiple workers to identify each worker.")
  flags.DEFINE_bool("transliterate", True,
                    help="Transliterate to hindi.")
  flags.DEFINE_bool("language_tag", True,
                    help="Use language special symbol.")
  flags.DEFINE_string("major_language", 'english',
                    help="Major document lang english/hindi.")
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(create_data)
