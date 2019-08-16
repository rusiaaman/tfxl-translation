# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata
import re
import six
from functools import partial


SPIECE_UNDERLINE = '▁'
START_CH = ord('\u0900')
END_CH = ord('\u097F')
SUBSTITUTES = {'\u0310':'\u0901','\u1CDA':'\u0951','\u1CDC':'\u0952','\u0953':'\u0300'
        ,'\u0954':'\u0301','\u0953':'\u0300'}


def is_hindi(character):
        oc = ord(character)
        if oc>=START_CH and oc<=END_CH:
                return True
        else:
                return False
def is_english(character):
    return re.match(r'[a-zA-Z]',character) is not None


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def print_(*args):
  new_args = []
  for arg in args:
    if isinstance(arg, list):
      s = [printable_text(i) for i in arg]
      s = ' '.join(s)
      new_args.append(s)
    else:
      new_args.append(printable_text(arg))
  print(*new_args)


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
  if remove_space:
    outputs = ' '.join(inputs.strip().split())
  else:
    outputs = inputs
  outputs = outputs.replace("``", '"').replace("''", '"').replace('”','"')\
          .replace("“",'"').replace("‘","'")

  if six.PY2 and isinstance(outputs, str):
    outputs = outputs.decode('utf-8')

  if not keep_accents:
    outputs = unicodedata.normalize('NFKD', outputs)
    outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
  # return_unicode is used only for py2

  # note(zhiliny): in some systems, sentencepiece only accepts str for py2
  if six.PY2 and isinstance(text, unicode):
    text = text.encode('utf-8')

  if not sample:
    pieces = sp_model.EncodeAsPieces(text)
  else:
    pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
  new_pieces = []
  for piece in pieces:
    if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
      cur_pieces = sp_model.EncodeAsPieces(
          piece[:-1].replace(SPIECE_UNDERLINE, ''))
      if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        if len(cur_pieces[0]) == 1:
          cur_pieces = cur_pieces[1:]
        else:
          cur_pieces[0] = cur_pieces[0][1:]
      cur_pieces.append(piece[-1])
      new_pieces.extend(cur_pieces)
    else:
      new_pieces.append(piece)

  # note(zhiliny): convert back to unicode for py2
  if six.PY2 and return_unicode:
    ret_pieces = []
    for piece in new_pieces:
      if isinstance(piece, str):
        piece = piece.decode('utf-8')
      ret_pieces.append(piece)
    new_pieces = ret_pieces

  return new_pieces


def split_on_language(text, major_language='english'):
  last_lang = False if major_language=='english' else True
  all_text = ['']
  langs = [last_lang]
  for t in text:
    lang = True if is_hindi(t) else False if is_english(t) else None
    if lang is None:
      all_text[-1]+=t
    elif lang==last_lang:
      all_text[-1]+=t
    else:
      all_text.append(t)
      langs.append(lang)
      last_lang = lang

  return all_text,(langs)

def encode_ids(sp, text, transliterate = False,
                  language_tag = False, eng_id = 0, hin_id = 0, major_language='english',
                  return_unicode = False, sample = False):
  
  if transliterate:
    from cltk.corpus.sanskrit.itrans.unicode_transliterate import ItransTransliterator as its
    assert eng_id is not None and hin_id is not None
    text = unicodedata.normalize("NFKC",text)
    for sub,tosub in SUBSTITUTES.items():
      text = text.replace(sub,tosub)
    dats,langs = split_on_language(text,major_language=major_language)
    trans = list(map(lambda x: its.to_itrans(x,'hi') if x.strip() else '',dats))
    enfn = partial(encode_pieces,return_unicode=return_unicode,
                      sample=sample)
    transpieces = list(map(lambda x: enfn(sp,x),trans))
    transids = [[sp.PieceToId(piece) for piece in pieces] for pieces in transpieces]

    if language_tag:
      langs = [hin_id if l else eng_id for l in langs]
      for i in range(len(transids)):
        transids[i].insert(0,langs[i])
    transids = [x for t in transids for x in t]
    return transids
  else:
    pieces =  encode_pieces(sp,text,return_unicode=return_unicode,
                      sample=sample)

    return [sp.PieceToId(piece) for piece in pieces]


if __name__ == '__main__':
  import sentencepiece as spm

  sp = spm.SentencePieceProcessor()
  sp.load('spiece.model')

  print_(u'I was born in 2000, and this is falsé.')
  print_(u'ORIGINAL', sp.EncodeAsPieces(u'I was born in 2000, and this is falsé.'))
  print_(u'OURS', encode_pieces(sp, u'I was born in 2000, and this is falsé.'))
  print(encode_ids(sp, u'I was born in 2000, and this is falsé.'))
  print_('')
  prepro_func = partial(preprocess_text, lower=True)
  print_(prepro_func('I was born in 2000, and this is falsé.'))
  print_('ORIGINAL', sp.EncodeAsPieces(prepro_func('I was born in 2000, and this is falsé.')))
  print_('OURS', encode_pieces(sp, prepro_func('I was born in 2000, and this is falsé.')))
  print(encode_ids(sp, prepro_func('I was born in 2000, and this is falsé.')))
  print_('')
  print_('I was born in 2000, and this is falsé.')
  print_('ORIGINAL', sp.EncodeAsPieces('I was born in 2000, and this is falsé.'))
  print_('OURS', encode_pieces(sp, 'I was born in 2000, and this is falsé.'))
  print(encode_ids(sp, 'I was born in 2000, and this is falsé.'))
  print_('')
  print_('I was born in 92000, and this is falsé.')
  print_('ORIGINAL', sp.EncodeAsPieces('I was born in 92000, and this is falsé.'))
  print_('OURS', encode_pieces(sp, 'I was born in 92000, and this is falsé.'))
  print(encode_ids(sp, 'I was born in 92000, and this is falsé.'))
  print(encode_ids(sp, 'I was born in 2000, and this is falsé.'))
  print_('')
  print_('समर्थक-वर्ग संक्षेपण १३०, छोड़ देना')
  print_(encode_pieces(sp, 'समर्थक-वर्ग संक्षेपण १३०, छोड़ देना'))
  print(encode_ids(sp, 'समर्थक-वर्ग संक्षेपण १३०, छोड़ देना'))

  print_('')
  print_('समर्थक-वर्ग संक्षेपण १३०, छोड़ देना')
  print_(encode_pieces(sp, 'समर्थक-वर्ग संक्षेपण १३०, छोड़ देना'))
  print(encode_ids(sp, 'समर्थक-वर्ग संक्षेपण १३०, छोड़ देना', transliterate=True,
                  language_tag=True, hin_id=7, eng_id=8))
  print(sp.decode_ids(encode_ids(sp, 'समर्थक-वर्ग संक्षेपण १३०, छोड़ देना', transliterate=True,
                  language_tag=True, hin_id=7, eng_id=8)))