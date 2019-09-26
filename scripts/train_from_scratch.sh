#!/bin/bash

# Path
GSDATA=gs://<your bucket>/<parallel-data>
GSEXP=gs://<your bucket>/translation-exp-hi-en

# TPU setting
NUM_HOST=1
NUM_CORE=8 # TPUv3

TEST_NUM_HOST=1
TEST_NUM_CORE=8 # TPUv3

# Model
N_LAYER=18
D_MODEL=1024
D_EMBED=1024
N_HEAD=16
D_HEAD=64
D_INNER=4096

# Training
SEQ_LEN=256
MEM_LEN=0
TGT_LEN=160
TRAIN_BSZ=256
VALID_BSZ=128

# Testing
TEST_SEQ_LEN=256
TEST_TGT_LEN=160
TEST_MEM_LEN=0
TEST_CLAMP_LEN=1000
TEST_BSZ=8

TRAIN_NUM_PASSES=15
TEST_NUM_PASSES=1

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --data_dir=${GSDATA}/tfrecords \
        --record_info_dir=${GSDATA}/tfrecords/ \
        --num_passes=$TRAIN_NUM_PASSES \
        --model_dir=${GSEXP} \
        --untie_r=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.3 \
        --dropatt=0.0 \
        --init_std=0.005 \
        --learning_rate=2.5e-4 \
        --warmup_steps=5000 \
        --train_steps=60000 \
        --tgt_len=${TGT_LEN} \
        --seq_len=${SEQ_LEN} \
        --mem_len=${MEM_LEN} \
        --same_length=False \
        --train_batch_size=${TRAIN_BSZ} \
        --num_hosts=${NUM_HOST} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=1000 \
        --save_steps=3000 \
        --use_tpu=True \
        --do_eval=False \
        --bi_mask=True \
        --nmt \
        --src_lang=english \
        --tgt_lang=hindi \
        ${@:2}

elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python train.py \
        --data_dir=${GSDATA}/tfrecords \
        --record_info_dir=${GSDATA}/tfrecords/ \
        --num_passes=$TEST_NUM_PASSES \
        --model_dir=${GSEXP} \
        --untie_r=True \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --tgt_len=${TEST_TGT_LEN} \
        --seq_len=${TEST_SEQ_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --train_steps=1500000 \
        --same_length=False \
        --eval_batch_size=${TEST_BSZ} \
        --num_host=${TEST_NUM_HOST} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --use_tpu=True \
        --do_train=False \
        --do_eval_only=True \
        --eval_split=dev \
        --bi_mask=True  \
        --nmt \
        --src_lang=english \
        --tgt_lang=hindi \
        ${@:2}

else
    echo 'unknown argment 1'
fi

