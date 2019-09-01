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


parser = argparse.ArgumentParser()
# Model
parser.add_argument("--tgt_len", default=70, type=int,
      help="Number of steps to predict")
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
                    help="Same length attention", action='store_true')
parser.add_argument("--tie_weight", type=bool, default=True,
      help="Tie embedding and softmax weight.")
# Data and memory
parser.add_argument("--n_token", default=32000, help='vocab size', type=int)
parser.add_argument("--batch_size", default=1, help='batch size', type=int)
parser.add_argument("--max_mem_length", default=128,
                    help="Max sequence length for cached hidden states"
                    " which each predicted token is conditioned upon"
                    ". Directly increases the memory requirement", type=int)
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
parser.add_argument("--num_samples", default=1,
                    help="Number of samples to predict per instance", type=int)
parser.add_argument(
    "--interactive",
    default=False,
    help="Flag for interactive prediction through command line",
    action='store_true')
parser.add_argument(
    "--unconditional",
    default=False,
    help="Prints samples wihtout any prompt",
    action='store_true')
parser.add_argument(
    "--top_p",
    default=0,
    help="Top-p coverage to use. Set 0 to use top_k sampling",
    type=float)
parser.add_argument(
    "--top_k",
    default=40,
    help="Top-k sampling strategy parameter. Use only when top-p is zero. Set"
    "-1 to use all the samples",
    type=int)
parser.add_argument("--temperature", default=1,
                    help="Scaling factor for logits", type=int)
parser.add_argument("--num_toks_pred", default=128,
                    help="Number of tokens to predict", type=int)
# Hindi specifics
parser.add_argument("--transliterate", action="store_true",
                  help="Transliterate to hindi.")
parser.add_argument("--language_tag", action="store_true",
                  help="Use language special symbol.")
parser.add_argument("--major_language", default='english',
                    help="Major document lang english/hindi.")

FLAGS = parser.parse_args()


def get_preprocessor(examples, tokenize_fn, pad_ids):
    """
    Input:
    examples: [List[str]] input texts
    tokenize_fn: [function] encodes text into IDs
    Output:
    tf input features
    """
    def generator():
        for example in examples:
            tokens = tokenize_fn(example)
            yield pad_ids + tokens

    return generator


def get_input_dataset(preprocessor):
    """Returns tf.data.Dataset for input"""
    batch_size = FLAGS.batch_size
    max_mem_length = FLAGS.max_mem_length

    def mask(ids):
        example = {'input': ids}
        input_k = example['input'][-max_mem_length:]
        seq_len = tf.shape(input_k)[0]
        input_mask = tf.tile(
            tf.convert_to_tensor(
                [0],
                dtype=tf.float32),
            [seq_len])
        pad_len = tf.maximum(0, max_mem_length - seq_len)
        pad_tensor = tf.concat([[[pad_len]], [[0]]], axis=-1)
        input_mask = tf.pad(input_mask, pad_tensor, constant_values=1)
        return example

    dataset = tf.data.Dataset.from_generator(preprocessor,
                                             output_types=tf.int32)
    dataset = dataset.map(mask)

    dataset = dataset.batch(batch_size,
                            drop_remainder=False)
    dataset.prefetch(1)
    return dataset


def get_logits(features,mems):
    """Builds the graph for calculating the final logits"""
    is_training = False

    cutoffs = []
    train_bin_sizes = []
    eval_bin_sizes = []
    proj_share_all_but_first = True
    n_token = FLAGS.n_token

    batch_size = FLAGS.batch_size

    inp = tf.transpose(features["input"], [1, 0])
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
        mem_len=FLAGS.max_mem_length,
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
        infer=True)

    return logits,new_mems

def sampling_strategy():
    """Based on flags return either top_k or
    top_p strategy."""
    if FLAGS.top_p != 0:
        return 'top_p'

    return 'top_k'


def sample_token(logits):
    """
    Inputs:
    logits: tf.Tensor([batch_size,len,num_tokens])
    Outpus:
    samples: tf.Tensor([batch_size,len])
    """
    # credits: https://github.com/nshepperd/gpt-2

    logits /= FLAGS.temperature

    batch_size = tf.shape(logits)[0]
    seq_len = tf.shape(logits)[1]
    num_toks = tf.shape(logits)[2]

    if sampling_strategy() == 'top_p':
        logits_sorted = tf.sort(logits,
                                direction="DESCENDING",
                                axis=-1)
        probs = tf.nn.softmax(logits_sorted, axis=-1)
        cum_probs = tf.math.cumsum(probs,
                                   axis=-1,
                                   exclusive=True)
        logits_masked = tf.where(cum_probs < FLAGS.top_p,
                                 logits_sorted,
                                 tf.ones_like(logits_sorted) * 100)
        min_logits = tf.reduce_min(logits_masked, axis=-1)

        logits_masked = tf.where(logits < min_logits,
                                 tf.ones_like(logits) * -1e10,
                                 logits)

    elif sampling_strategy() == "top_k":
        if FLAGS.top_k != 0:
            values, _ = tf.nn.top_k(logits, k=FLAGS.top_k)
            min_values = values[:, :, -1:]
            logits_masked = tf.where(
                logits < min_values,
                tf.ones_like(logits, dtype=logits.dtype) * -1e10,
                logits,
            )
    else:
        raise NotImplementedError("Invalid sampling strategy")

    logits_masked = tf.reshape(logits_masked, (-1, num_toks))

    samples = tf.random.categorical(logits_masked,
                                    num_samples=1,
                                    dtype=tf.int32)

    probs = tf.nn.softmax(tf.reshape(logits, (-1, num_toks)), axis=-1)
    confidences = tf.gather_nd(params=probs, batch_dims=1, indices=samples)

    return tf.reshape(samples, (batch_size, seq_len)),\
        tf.reshape(confidences, (batch_size, seq_len))

def prediction_graph():
    """Gets features and
    return predicted tokens)
    features: Dict[str:tf.train.features] Contains following features:
              input_k
              seg_id
              input_mask
    """



    features = {
        "input": tf.placeholder(tf.int32, (None, None))
    }

    # Calculating hidden states of inputs and getting latest logit
    logits,mems = get_logits(features,mems=None)
    batch_size = tf.shape(mems[0])[1]
    logits = tf.reshape(logits,(batch_size,1,-1))
    latest_toks,latest_confs = sample_token(logits) 
    all_confs = latest_confs
    all_toks = latest_toks

    def cond(*_):
        """Dummy condition since we stop based on iteration"""
        return True
    def body(mems, latest_toks, all_toks, all_confs):
        """The main body of sampling loop.
        """

        features = {  
          'input': latest_toks
        }
        logits,mems = get_logits(features,mems=mems)
        logits = tf.reshape(logits,(batch_size,1,-1))
        latest_toks,latest_confs = sample_token(logits) 
        all_confs = tf.concat([all_confs,latest_confs],axis=-1)
        all_toks = tf.concat([all_toks,latest_toks],axis=-1)

        return [mems, latest_toks, all_toks, all_confs]

    args = tf.while_loop(
        cond=cond,
        body=body,
        maximum_iterations=FLAGS.num_toks_pred - 2,
        loop_vars=[mems, latest_toks, all_toks, all_confs],
        shape_invariants=[
            [tf.TensorShape([None, None, None]) for _ in range(len(mems))],
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None])
        ]
    )

    predicted_tokens, predicted_confs = args[-2:]
    return (predicted_tokens, predicted_confs), features


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
                           transliterate=FLAGS.transliterate, language_tag=FLAGS.language_tag,
                           eng_id=ENG_ID, hin_id=HIN_ID, major_language=FLAGS.major_language)
        return text

    # Temporary fix for context problem.
    if FLAGS.major_language=="hindi":
      pad_txt = "कालिंजर दुर्ग, भारतीय राज्य उत्तर प्रदेश के बांदा जिला स्थित एक दुर्ग है। बुन्देलखण्ड क्षेत्र में विंध्य पर्वत पर स्थित यह दुर्ग विश्व धरोहर स्थल खजुराहो से ९७.७ किमी दूर है। इसे भारत के सबसे विशाल और अपराजेय दुर्गों में गिना जाता रहा है। इस दुर्ग में कई प्राचीन मन्दिर हैं। इनमें कई मंदिर तीसरी से पाँचवीं सदी गुप्तकाल के हैं। यहाँ के शिव मन्दिर के बारे में मान्यता है कि सागर-मन्थन से निकले कालकूट विष को पीने के बाद भगवान शिव ने यही तपस्या कर उसकी ज्वाला शांत की थी। कार्तिक पूर्णिमा के अवसर पर लगने वाला कतिकी मेला यहाँ का प्रसिद्ध सांस्कृतिक उत्सव है। भारत की स्वतंत्रता के पश्चात इसकी पहचान एक महत्वपूर्ण ऐतिहासिक धरोहर के रूप में की गयी है। वर्तमान में यह दुर्ग भारतीय पुरातत्त्व सर्वेक्षण विभाग के अधिकार एवं अनुरक्षण में है।"
      pad_ids = tokenize_fn(pad_txt)
      pad_ids = pad_ids+[EOP_ID]
    else:
      pad_txt = """In 1991, the remains of Russian Tsar Nicholas II and his family
                (except for Alexei and Maria) are discovered.
                The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
                remainder of the story. 1883 Western Siberia,
                a young Grigori Rasputin is asked by his father and a group of men to perform magic.
                Rasputin has a vision and denounces one of the men as a horse thief. Although his
                father initially slaps him for making such an accusation, Rasputin watches as the
                man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
                the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
                 with people, even a bishop, begging for his blessing. """
      pad_ids = tokenize_fn(pad_txt)
      pad_ids = pad_ids+[EOD_ID]


    to_special_symbol = {v:k for k,v in special_symbols.items()}
    def parse_ids(toks):
        """Uses sentencepiece to conver to text. Subsitute
        EOP_ID and EOD_ID with new lines, and rest with their names"""
        all_sents = []
        all_langs = []
        lang = 0 if FLAGS.major_language=='english' else 1
        indices_langs = [(-1,lang)]+[(i,int(toks[i]==HIN_ID))
                         for i in range(len(toks)) if toks[i] in [ENG_ID,HIN_ID]]
        indices = indices_langs + [(len(toks),None)]

        all_sents = [toks[indices[i][0]+1:indices[i+1][0]] for i in range(len(indices)-1)]
        all_langs = [ind[1] for ind in indices[:-1]] 
        
        all_sents = list(map(sp.decode_ids,all_sents))
        if FLAGS.transliterate:
          all_sents = [transliterate_back(_sent,_lang) for _sent,_lang in zip(all_sents,all_langs)]
        
        sent = ''.join(all_sents)

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
                                            tokenize_fn, pad_ids)
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

        if FLAGS.unconditional or FLAGS.interactive:
            tf.logging.info("Interactive flag received."
                            " Ignoring input files if any.")
            while True:
                if FLAGS.unconditional:
                    text = ""
                else:
                    text = input("----PROMPT----\n")
                outputs = predict([text] * FLAGS.num_samples)
                for i, (output,_) in enumerate(outputs):
                    out = parse_ids(output.tolist())
                    print("======SAMPLE {}======".format(i))
                    print(out)
                    print("=====================")
                if FLAGS.unconditional:
                    break
        else:
            assert FLAGS.input_file!="", "Please provide either an"\
            " input file or set interactive flag for command line input"
            assert os.path.exists(FLAGS.input_file), FLAGS.input_file+\
            " does not exists"

            with open(FLAGS.input_file) as f:
                texts = []
                text = ""
                for line in f:
                    if line.strip()=="":
                        if text!="":
                            # Removing the last <eop> of prompt
                            # since it is not desired
                            if text.endswith("<eop>"):
                                text=text[:-5]
                            texts.extend([text]*FLAGS.num_samples)
                            text=""
                        continue
                    text+=re.sub(r'\n','<eop>',line)
                if text!="":
                    texts.extend([text]*FLAGS.num_samples)

            tf.logging.info("Got %s lines in the input file",
                            len(texts)//FLAGS.num_samples)
            tf.logging.info("Sampling each line %s times",FLAGS.num_samples)

            outputs = iter(predict(texts))
            with open(os.path.join(FLAGS.input_file+".xlnet"),'w') as f:
                for i in range(0,len(texts),FLAGS.num_samples):
                    f.write("\n======Example {}=================\n".format(i))
                    f.write(texts[i])
                    for j in range(FLAGS.num_samples):
                        output,_ = next(outputs)
                        out = parse_ids(output.tolist())
                        f.write("\n======Example {} SAMPLE {}======\n".format(i,j))
                        f.write(out)
                        f.write("\n==================================\n")

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
