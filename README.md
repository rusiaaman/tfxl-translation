# About

This project builds a translation system between hindi and english. However, it easily can be adapted to other language pairs with a few changes. The model architecture is a transformer-decoder built on transformer-xl code and implements a novel training procedure.

The research was supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)

# Hindi translation implementation
We transliterate hindi to english using ITRANS encoding and train `sentencepiece` encoder on only latin characters thus obtained. However, hindi unicode characters can also be used with this implementation.

IITB parallel corpus was used as a data source for training. Pretrained model will be released soon.

# Usage

## Data preparation
### Creating tf records
```
python data_utils_nmt.py --use_tpu --bsz_per_host 32 --seq_len 128  --tgt_len 64 --bi_data=False \
--src_file test.hi --tgt_file test.en --src_lang hindi --tgt_lang english --long_sentences truncate \
--sp_path spiece.model --save_dir save-3 --split train \
--transliterate --language_tag --use_sos
```

## Training
### Tranining on TPUv3
Training example script is available in `scripts` folder: `train_from_scratch.sh` which can be used to train the translation system from scratch.

### Training on GPU
*Coming Soon*

## Inference
Interactive or complete text file translations can be done using `translate.py`. 
```
#!/bin/bash

# Path
GSEXP=$1


# Model
N_LAYER=18
D_MODEL=1024
D_EMBED=1024
N_HEAD=16
D_HEAD=64
D_INNER=4096
# Testing
TEST_SEQ_LEN=256
TEST_TGT_LEN=128
TEST_MEM_LEN=0
TEST_CLAMP_LEN=1000
TEST_BSZ=4


python translate.py \
        --untie_r \
        --n_layer=$N_LAYER \
        --d_model=$D_MODEL \
        --d_embed=$D_EMBED \
        --n_head=$N_HEAD \
        --d_head=$D_HEAD \
        --d_inner=$D_INNER \
        --seq_len=$TEST_SEQ_LEN \
        --clamp_len=$TEST_CLAMP_LEN \
        --batch_size=$TEST_BSZ \
        --spiece_model_file=spiece-iitb-hi-en.model \
        --bi_mask  \
        --use_sos \
        --src_lang=english \
        --tgt_lang=hindi \
        --max_decode_length=$TEST_TGT_LEN \
        --transliterate \
        --beam_size=8\
        --beam_alpha=0.8\
        ${@:2}

```


# Todo
- [] Fix relative positional encoding for bidirectional attention case
- [] Release pretrained models
- [] Add translation samples
- [] Add gpu training
- [] Improve documentation
