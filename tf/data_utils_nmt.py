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
SOS_ID = special_symbols["<s>"]
EOS_ID = special_symbols["</s>"]
PAD_ID = special_symbols["<pad>"]

def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))



def format_filename_gen(prefix, seq_len, bi_data, suffix,
                    src_lang,tgt_lang,uncased=False,):
  """docs."""

  if not uncased:
    uncased_str = ""
  else:
    uncased_str = "uncased."
  if bi_data:
    bi_data_str = "bi"
  else:
    bi_data_str = "uni"


  file_name = "{}-{}_{}.seqlen-{}.{}{}.gen.{}".format(
      src_lang[:2],tgt_lang[:2],
      prefix, seq_len, uncased_str, 
      bi_data_str, suffix)

  return file_name

def _create_data(idx, src_file, tgt_file, src_lang, tgt_lang,
                  transliterate=True, language_tag=True):
  # Load sentence-piece model
  sp = spm.SentencePieceProcessor()
  sp.Load(FLAGS.sp_path)

  input_data = []
  target_data = []
  target_mask_data = []
  input_mask_data = []
  total_line_cnt = 0
  for src_line,tgt_line in zip(tf.gfile.Open(src_file),
                                tf.gfile.Open(tgt_file)):
    if total_line_cnt % 100000 == 0:
      tf.logging.info("Loading line %d", total_line_cnt)

    if not src_line.strip() or not tgt_line.strip():
      continue

    if FLAGS.from_raw_text:
      src_sent = preprocess_text(src_line.strip(), lower=FLAGS.uncased)
      tgt_sent = preprocess_text(tgt_line.strip(), lower=FLAGS.uncased)
      src_sent = encode_ids(sp, src_sent,
                           transliterate=transliterate, language_tag=False)
      tgt_sent = encode_ids(sp, tgt_sent,
                           transliterate=transliterate, language_tag=False)
      tgt_sent = tgt_sent+[EOS_ID]
      tgt_sent_input = tgt_sent[:-1]
      tgt_sent_output = tgt_sent[1:]

      #Maximum size allowed for target
      tgt_sent_output = tgt_sent_output[:FLAGS.tgt_len]
      tgt_sent_input  = tgt_sent_input[:FLAGS.tgt_len]

      if FLAGS.language_tag:
        src_id = ENG_ID if src_lang=="english" else HIN_ID
        tgt_id = ENG_ID if tgt_lang=="english" else HIN_ID
        src_sent_e = [src_id]+src_sent
        tgt_sent_input = [tgt_id]+tgt_sent_input

      if FLAGS.use_sos:
        src_sent_e = [SOS_ID]+src_sent_e
        tgt_sent_input = [SOS_ID]+tgt_sent_input

      input_len = len(src_sent_e)+len(tgt_sent_input)+1 #One extra for EOS after source
      if input_len>FLAGS.seq_len:
        if FLAGS.long_sentences=='ignore':
          continue
        else:
          # Truncate in ratio of their original lenghts
          to_trunc = input_len - FLAGS.seq_len
          len_ratio = len(src_sent_e)/len(tgt_sent_input)
          to_trunc_src = min(int(len_ratio*to_trunc),to_trunc)
          to_trunc_tgt = to_trunc-to_trunc_src
          if to_trunc_src>0:
            src_sent_e = src_sent_e[:-to_trunc_src]
          if to_trunc_tgt>0:
            tgt_sent_input = tgt_sent_input[:-to_trunc_tgt]
            tgt_sent_output = tgt_sent_output[:-to_trunc_tgt]
          input_len = FLAGS.seq_len
          assert len(src_sent_e)+len(tgt_sent_input)+1 == input_len

      # Target padding to tgt_len on the left side
      target_mask = [0]*(FLAGS.tgt_len-len(tgt_sent_output))+ [1]*len(tgt_sent_output)
      target = [PAD_ID]*(FLAGS.tgt_len-len(tgt_sent_output))+ tgt_sent_output

      # Paddings for input 
      pads = [PAD_ID]*(FLAGS.seq_len-input_len)
      instance = pads+src_sent_e+[EOS_ID]+tgt_sent_input
      input_mask = [0]*len(pads)+[1]*(len(instance)-len(pads))


      assert len(instance) == FLAGS.seq_len, len(instance)
      assert len(input_mask) == FLAGS.seq_len, len(input_mask)
      assert len(target) == FLAGS.tgt_len, len(target)
      assert len(target_mask) == FLAGS.tgt_len, len(target_mask)
    else:
      raise Exception("Loading from id files not yet supported")

    input_data.append(np.array(instance,dtype=np.int64))
    target_data.append(np.array(target,dtype=np.int64))
    target_mask_data.append(np.array(target_mask,dtype=np.int64))
    input_mask_data.append(np.array(input_mask,dtype=np.int64))
    total_line_cnt+=1

  tf.logging.info("Finish with line %d", total_line_cnt)
  if total_line_cnt == 0:
    raise Exception("Files have no valid data")

  tf.logging.info("[Task %d] Total number line: %d", idx, total_line_cnt)

  tfrecord_dir = os.path.join(FLAGS.save_dir, "tfrecords")

  file_name, num_batch = create_tfrecords(
      save_dir=tfrecord_dir,
      basename="{}-{}-{}".format(FLAGS.split, idx, FLAGS.pass_id),
      data=(input_data,target_data,target_mask_data,input_mask_data),
      seq_len=FLAGS.seq_len,
      bi_data=FLAGS.bi_data,
      sp=sp
  )

  record_info = {
      "filenames": [file_name],
      "langs": [src_lang,tgt_lang],
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

  if FLAGS.tgt_len is None:
    FLAGS.tgt_len = FLAGS.seq_len//2 

  # Create and dump corpus_info from task 0
  if FLAGS.task == 0:
    corpus_info = {
        "vocab_size": VOCAB_SIZE,
        "bsz_per_host": FLAGS.bsz_per_host,
        "num_core_per_host": FLAGS.num_core_per_host,
        "seq_len": FLAGS.seq_len,
        "uncased": FLAGS.uncased,
        "bi_data": FLAGS.bi_data,
        "use_sos": FLAGS.use_sos,
        "sp_path": FLAGS.sp_path,
        "src_file": FLAGS.src_file,
        "tft_file": FLAGS.tgt_file,
        "src_lang": FLAGS.src_lang,
        "tgt_lang": FLAGS.tgt_lang,
    }
    corpus_info_path = os.path.join(FLAGS.save_dir, "corpus_info.json")
    with tf.gfile.Open(corpus_info_path, "w") as fp:
      json.dump(corpus_info, fp)

  # Interleavely split the work into FLAGS.num_task splits
  assert tf.gfile.Exists(FLAGS.src_file), f"{FLAGS.src_file} not found"
  assert tf.gfile.Exists(FLAGS.tgt_file), f"{FLAGS.tgt_file} not found"

  record_info = _create_data(FLAGS.task, FLAGS.src_file, FLAGS.tgt_file,
                             FLAGS.src_lang,
                             FLAGS.tgt_lang, 
                             transliterate=FLAGS.transliterate, 
                             language_tag=FLAGS.language_tag)

  record_prefix = "record_info-{}-{}-{}".format(
      FLAGS.split, FLAGS.task, FLAGS.pass_id)
  record_name = format_filename_gen(
      prefix=record_prefix,
      seq_len=FLAGS.seq_len,
      bi_data=FLAGS.bi_data,
      suffix="json",
      uncased=FLAGS.uncased,
      src_lang=FLAGS.src_lang,
      tgt_lang=FLAGS.tgt_lang)
  record_info_path = os.path.join(tfrecord_dir, record_name)

  with tf.gfile.Open(record_info_path, "w") as fp:
    json.dump(record_info, fp)



def create_tfrecords(save_dir, basename, data, seq_len,
                     bi_data, sp):
  input_data,target_data,target_mask_data,input_mask_data = data

  if bi_data:
    raise Exception("Bi directional data not supported right now")
  
  file_name = format_filename_gen(
      prefix=basename,
      seq_len=seq_len,
      bi_data=bi_data,
      suffix="tfrecords",
      uncased=FLAGS.uncased,
      src_lang=FLAGS.src_lang,
      tgt_lang=FLAGS.tgt_lang)

  save_path = os.path.join(save_dir, file_name)
  record_writer = tf.python_io.TFRecordWriter(save_path)
  tf.logging.info("Start writing %s.", save_path)

  num_batch = 0

  for inputs,targets,inp_masks,tgt_masks in zip(input_data,target_data,input_mask_data,target_mask_data):
    feature = {
        "input": _int64_feature(inputs),
        "labels": _int64_feature(targets),
        "input_mask": _float_feature(inp_masks),
        "target_mask": _float_feature(tgt_masks),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    record_writer.write(example.SerializeToString())
    num_batch += 1

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

  dataset = tf.data.TFRecordDataset(dataset)

  # (zihang): since we are doing online preprocessing, the parsed result of
  # the same input at each time will be different. Thus, cache processed data
  # is not helpful. It will use a lot of memory and lead to contrainer OOM.
  # So, change to cache non-parsed raw data instead.
  
  if not toeval:
    dataset = dataset.cache().shuffle(10000).repeat().map(parser)
  else:
    dataset = dataset.map(parser)

  dataset = dataset.batch(bsz_per_core, drop_remainder=True)
  dataset = dataset.prefetch(num_core_per_host * bsz_per_core)

  return dataset

def get_dataset(params, num_hosts, num_core_per_host, split, file_names,
                num_batch, seq_len, use_bfloat16=False, toeval=True, tgt_len=None):

  bsz_per_core = params["batch_size"]
  if num_hosts > 1:
    host_id = params["context"].current_host
  else:
    host_id = 0

  if tgt_len is None:
    tgt_len = seq_len//2
  #### Function used to parse tfrecord
  def parser(record):
    """function used to parse tfrecord."""
    record_spec = {
        "input": tf.FixedLenFeature([seq_len], tf.int64),
        "labels": tf.FixedLenFeature([tgt_len], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_len],tf.float32),
        "target_mask": tf.FixedLenFeature([tgt_len],tf.float32)
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
    src_lang,
    tgt_lang,
    bsz_per_host,
    seq_len,
    bi_data,
    num_hosts=1,
    num_core_per_host=1,
    uncased=False,
    num_passes=None,
    use_bfloat16=False,
    toeval=False,
    tgt_len=None):

  # Merge all record infos into a single one
  record_glob_base = format_filename_gen(
      prefix="record_info-{}-*".format(split),
      seq_len=seq_len,
      bi_data=bi_data,
      suffix="json",
      uncased=uncased,
      src_lang=src_lang,
      tgt_lang=tgt_lang)

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
        toeval=toeval,
        tgt_len=tgt_len)

    return dataset

  return input_fn, record_info


if __name__ == "__main__":
  FLAGS = flags.FLAGS
  flags.DEFINE_bool("use_tpu", True, help="whether to use TPUs")
  flags.DEFINE_integer("bsz_per_host", 32, help="batch size per host.")
  flags.DEFINE_integer("num_core_per_host", 8, help="num TPU cores per host.")
  flags.DEFINE_integer("seq_len", 512,
                       help="Sequence length.")
  flags.DEFINE_integer("tgt_len", None,
                       help="Targets will be padded to this size. Default is seq_len//2")
  flags.DEFINE_bool("uncased", False, help="Use uncased inputs or not.")
  flags.DEFINE_bool("bi_data", True,
                    help="whether to create bidirectional data")
  flags.DEFINE_bool("use_sos", True,
                    help="whether to use SOS.")
  flags.DEFINE_bool("from_raw_text", True,
                    help="Whether the input is raw text or encoded ids.")
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
  flags.DEFINE_string("src_file", 'IITB.en-hi.hi',
                    help="Source language file.")
  flags.DEFINE_string("tgt_file", 'IITB.en-hi.en',
                    help="Target language file.")
  flags.DEFINE_string("src_lang", 'hindi',
                    help="Source language file.")
  flags.DEFINE_string("tgt_lang", 'english',
                    help="Target language file.")
  flags.DEFINE_enum("long_sentences", 'truncate', ['truncate','ignore'],
                    help="Whether .")
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(create_data)
