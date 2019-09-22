"""Generate language using XLNet"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import re

from tqdm import tqdm
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf
import sentencepiece as spm
import collections

from prepro_utils import preprocess_text, encode_ids
import model
import beam_search

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

SOS_ID = special_symbols['<s>']
EOS_ID = special_symbols['</s>']
UNK_ID = special_symbols["<unk>"]
EOD_ID = special_symbols["<eod>"]
EOP_ID = special_symbols["<eop>"]
HIN_ID = special_symbols["<hi>"]
ENG_ID = special_symbols["<eng>"]
PAD_ID = special_symbols["<pad>"]

parser = argparse.ArgumentParser()
# Model
parser.add_argument("--n_layer", default=6, type=int,
      help="Number of layers.")
parser.add_argument("--d_model", default=500, type=int,
      help="Dimension of the model.")
parser.add_argument("--d_embed", default=500, type=int,
      help="Dimension of the embeddings.")
parser.add_argument("--n_head", default=10, type=int,
      help="Number of attention heads.")
parser.add_argument("--d_head", default=50, type=int,
      help="Dimension of each attention head.")
parser.add_argument("--d_inner", default=1000, type=int,
      help="Dimension of inner hidden size in positionwise feed-forward.")
parser.add_argument("--untie_r", action='store_true',
      help="untie r_w_bias and r_r_bias")
parser.add_argument("--clamp_len", default=-1,
                    help="Clamp length", type=int)
parser.add_argument("--same_length", default=False,
                    help="Same length attention")
parser.add_argument("--tie_weight", type=bool, default=True,
      help="Tie embedding and softmax weight.")
# Data and memory
parser.add_argument("--seq_len", default=70,
      help="Maxmium number of steps in the input", type=int)
parser.add_argument("--n_token", default=32000, help='vocab size', type=int)
parser.add_argument("--batch_size", default=1, help='batch size', type=int)
parser.add_argument("--uncased", default=False,
                    help="Use uncased inputs or not.", type=bool)

# I/O paths
parser.add_argument("--init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                    "Could be a pretrained model or a finetuned model.")
parser.add_argument("--spiece_model_file", default="",
                    help="Sentence Piece model path.")
parser.add_argument("--input_file", default="",
                    help="File containing prompts separated by empty new line "
                    "for conditional sampling")

# prediction
parser.add_argument(
    "--interactive",
    default=False,
    help="Flag for interactive prediction through command line",
    action='store_true')
parser.add_argument("--beam_size",default=4,type=int,
    help="Beam width for beam search decoding")
parser.add_argument("--beam_alpha",default=0.6,type=float,
    help="alpha parameter for beam search decoding")
parser.add_argument("--max_decode_length", default=1024,
                    help="Maximum Number of tokens to predict", type=int)

# NMT specifics
parser.add_argument("--bi_mask",action="store_true",
                     help="Use bidirectional mask for source tokens")
parser.add_argument("--use_sos", default=True, type=bool,
                    help="whether to use SOS.")
parser.add_argument("--transliterate", action="store_true",
                  help="Transliterate to hindi.")
parser.add_argument("--src_lang", default='english',
                    help="Source lang english/hindi.")
parser.add_argument("--tgt_lang", default='hindi',
                    help="Target lang english/hindi.")

FLAGS = parser.parse_args()


def get_preprocessor(examples, tokenize_fn):
    """
    Input:
    examples: [List[str]] input texts
    tokenize_fn: [function] encodes text into IDs
    Output:
    tf input features
    """
    def generator():
        for i in range(0,len(examples),FLAGS.batch_size):
            batched = examples[i:i+FLAGS.batch_size]
            tokens_batched = list(map(tokenize_fn,batched))
            maxlen = max(map(len,tokens_batched))
            for tokens in tokens_batched:
                pad_len = maxlen-len(tokens)

                src_id = ENG_ID if FLAGS.src_lang=="english" else HIN_ID
                src_id = [src_id]
                if FLAGS.use_sos:
                    src_id = [SOS_ID] + src_id
                ids = src_id + tokens + [EOS_ID]

                if FLAGS.use_sos:
                    ids = ids + [SOS_ID]
                
                masks = [0]*pad_len+[1]*len(ids)
                ids = [PAD_ID]*pad_len+ids

                ids = ids[-FLAGS.seq_len:]
                masks = masks[-FLAGS.seq_len:]

                yield {'input':ids,'input_mask':masks}

    return generator


def get_input_dataset(preprocessor):
    """Returns tf.data.Dataset for input"""
    batch_size = FLAGS.batch_size

    dataset = tf.data.Dataset.from_generator(preprocessor,
                                             output_types={'input':tf.int32,
                                             'input_mask':tf.float32})
    dataset = dataset.batch(batch_size,
                            drop_remainder=False)
    dataset.prefetch(1)
    return dataset


def get_logits(input_ids,mems,input_mask,target_mask):
    """Builds the graph for calculating the final logits"""
    is_training = False

    cutoffs = []
    train_bin_sizes = []
    eval_bin_sizes = []
    proj_share_all_but_first = True
    n_token = FLAGS.n_token

    batch_size = FLAGS.batch_size

    features = {"input": input_ids}
    inp = tf.transpose(features["input"], [1, 0])
    input_mask = tf.transpose(input_mask, [1, 0])
    target_mask = tf.transpose(target_mask, [1, 0])
    tgt = None

    inp_perms, tgt_perms, head_tgt = None, None, None

    if FLAGS.init == "uniform":
      initializer = tf.initializers.random_uniform(
          minval=-FLAGS.init_range,
          maxval=FLAGS.init_range,
          seed=None)
    elif FLAGS.init == "normal":
      initializer = tf.initializers.random_normal(
          stddev=FLAGS.init_std,
          seed=None)
      proj_initializer = tf.initializers.random_normal(
          stddev=FLAGS.proj_init_std,
          seed=None)

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if proj_share_all_but_first:
      for i in range(1, len(tie_projs)):
        tie_projs[i] = True

    tf.logging.info("Vocab size : {}".format(n_token))
    tf.logging.info("Batch size : {}".format(batch_size))

    logits, new_mems = model.transformer(
        dec_inp=inp,
        target=tgt,
        mems=mems,
        n_token=n_token,
        n_layer=FLAGS.n_layer,
        d_model=FLAGS.d_model,
        d_embed=FLAGS.d_embed,
        n_head=FLAGS.n_head,
        d_head=FLAGS.d_head,
        d_inner=FLAGS.d_inner,
        dropout=0,
        dropatt=0,
        initializer=initializer,
        is_training=is_training,
        mem_len=FLAGS.seq_len+FLAGS.max_decode_length,
        cutoffs=cutoffs,
        div_val=1,
        tie_projs=tie_projs,
        input_perms=inp_perms,
        target_perms=tgt_perms,
        head_target=head_tgt,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        use_tpu=FLAGS.use_tpu,
        untie_r=FLAGS.untie_r,
        proj_same_dim=True,
        bidirectional_mask=FLAGS.bi_mask,
        infer=True,
        target_mask=target_mask,
        input_mask=input_mask,
        tgt_len=1)

    return logits,new_mems


def prediction_graph():
    """Gets features and
    return predicted tokens)
    features: Dict[str:tf.train.features] Contains following features:
              input_k
              seg_id
              input_mask
    """


    features = {
        "input": tf.placeholder(tf.int32, (None, None)),
        "input_mask":  tf.placeholder(tf.float32, (None, None))
    }
    batch_size = tf.shape(features['input'])[0]
    input_tensor = features['input']

    # Calculating hidden states of inputs and getting latest logit
    input_mask = features['input_mask']
    target_mask = tf.ones((tf.shape(input_tensor)[0],1))
    _,mems = get_logits(input_tensor,mems=None,input_mask=input_mask,
                             target_mask=target_mask)
    # logits = tf.reshape(logits,(batch_size,1,-1))
    # latest_toks,latest_confs = sample_token(logits) 
    # all_confs = latest_confs
    # all_toks = latest_toks

    def symbols_to_logits_fn(toks,_,mems):
        # We need only last token
        toks = toks[:,-1:]
        # input_mask set all the inputs to be valid
        input_mask = tf.ones_like(toks,dtype=tf.float32)
        # target_mask set to be of ones
        target_mask = tf.ones((tf.shape(toks)[0],1),dtype=tf.float32)
        mems = [tf.transpose(mems[i],[1,0,2]) if i<len(mems)-1 else \
                tf.transpose(mems[i],[1,0])
                for i in range(len(mems))]
        logits,mems = get_logits(toks,mems=mems,input_mask=input_mask,
                                         target_mask=target_mask)
        return logits,{i:tf.transpose(mems[i],[1,0,2]) if i<len(mems)-1 else \
                      tf.transpose(mems[i],[1,0])
                      for i in range(len(mems))}
    
    lang_id = ENG_ID if FLAGS.tgt_lang=="english" else HIN_ID
    initial_ids = tf.ones((batch_size),dtype=tf.int32)*lang_id

    mems = {i:tf.transpose(mems[i],[1,0,2]) if i<len(mems)-1 else \
              tf.transpose(mems[i],[1,0])
              for i in range(len(mems))}

    decoded_ids, scores = beam_search.sequence_beam_search(
    symbols_to_logits_fn, initial_ids, mems, FLAGS.n_token, FLAGS.beam_size,
    FLAGS.beam_alpha, FLAGS.max_decode_length, EOS_ID)
    top_decoded_ids = decoded_ids[:, 0, 1:]
    top_scores = scores[:, 0]
    return (top_decoded_ids, top_scores), features


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    # tf.logging.info('original name: %s', name)
    if name not in name_to_variable:
      continue
    # assignment_map[name] = name
    assignment_map[name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)

def init_from_checkpoint(FLAGS, global_vars=False):
  tvars = tf.global_variables() if global_vars else tf.trainable_variables()
  initialized_variable_names = {}
  if FLAGS.init_checkpoint is not None:
    if FLAGS.init_checkpoint.endswith("latest"):
      ckpt_dir = os.path.dirname(FLAGS.init_checkpoint)
      init_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    else:
      init_checkpoint = FLAGS.init_checkpoint

    tf.logging.info("Initialize from the ckpt {}".format(init_checkpoint))

    (assignment_map, initialized_variable_names
    ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Log customized initialization
    tf.logging.info("**** Global Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

def transliterate_back(text,lang):
  # English return as it is
  if text=="":
    return text
  if lang==0:
    return text
  from cltk.corpus.sanskrit.itrans.unicode_transliterate import ItransTransliterator as its
  return its.from_itrans(text,'hi')


def main():
    """Main function routine"""

    tf.logging.set_verbosity(tf.logging.INFO)

    # Text encoding
    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.spiece_model_file)

    def tokenize_fn(text):
        text = preprocess_text(text, lower=FLAGS.uncased)
        text = encode_ids(sp, text,
                           transliterate=FLAGS.transliterate, language_tag=False)
        return text


    to_special_symbol = {v:k for k,v in special_symbols.items()}
    def parse_ids(toks):
        """Uses sentencepiece to conver to text. Subsitute
        EOP_ID and EOD_ID with new lines, and rest with their names"""
        
        # IF EOS_ID was encountered rest will be pad ids
        print(toks)
        if EOS_ID in toks:
            toks = toks[:toks.index(EOS_ID)]

        sent = sp.decode_ids(toks)
        if FLAGS.transliterate and FLAGS.tgt_lang!='english':
          sent = transliterate_back(sent,FLAGS.tgt_lang)


        return sent

    predictions, features = prediction_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    init_from_checkpoint(FLAGS, global_vars=False)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        def predict(examples):
            """Given a list of texts in examples
            return the result"""
            preprocessor = get_preprocessor(examples,
                                            tokenize_fn)
            dataset = get_input_dataset(preprocessor)
            example = dataset.make_one_shot_iterator().get_next()

            num_examples = len(examples)
            num_batches = int(np.ceil(num_examples / FLAGS.batch_size))

            for _ in tqdm(range(num_batches)):
                inputs = sess.run(example)
                output, conf = sess.run(
                    predictions, feed_dict={
                        features[k]: v for k, v in inputs.items()})
                for _output,_conf in zip(output,conf):
                    yield _output,_conf

        if FLAGS.interactive:
            tf.logging.info("Interactive flag received."
                            " Ignoring input files if any.")
            while True:
                text = input("----PROMPT----\n")
                outputs = predict([text])
                output = next(outputs)
                out = parse_ids(output[0].tolist())
                print("======Translation======")
                print(out)
                print("=====================")
        else:
            assert FLAGS.input_file!="", "Please provide either an"\
            " input file or set interactive flag for command line input"
            assert os.path.exists(FLAGS.input_file), FLAGS.input_file+\
            " does not exists"

            with open(FLAGS.input_file) as f:
                texts = []
                for line in f:
                    texts.append(line.strip())

            tf.logging.info("Got %s lines in the input file",
                            len(texts))
            outputs = predict(texts)
            with open(os.path.join(FLAGS.input_file+".xlnet"),'w') as f:
                for i in range(0,len(texts)):
                    output,_ = next(outputs)
                    out = parse_ids(output.tolist())
                    f.write(out+'\n')
# Fixed flags
FLAGS.use_tpu = False
FLAGS.use_bfloat16 = False
FLAGS.dropout = 0
FLAGS.dropatt = 0
FLAGS.init = "normal"
FLAGS.init_std = 0.02
FLAGS.init_range = 0.1
FLAGS.proj_init_std = 0.01

if __name__ == "__main__":

    
    print("Args: {}".format(vars(FLAGS)))
    main()
