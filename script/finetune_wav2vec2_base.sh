#!/bin/bash

# ========= 設定 ========= #
DATA_DIR=/home/poyu39/github/poyu39/emotion-conformer/dataset/finetune/
SAVE_DIR=/home/poyu39/github/poyu39/emotion-conformer/model/wav2vec2-base/librispeech_iemocap/checkpoints
CHECKPOINT_PATH=/home/poyu39/github/poyu39/emotion-conformer/model/wav2vec2-base/librispeech/wav2vec_small.pt
TENSORBOARD_DIR=/home/poyu39/github/poyu39/emotion-conformer/model/wav2vec2-base/librispeech_iemocap/tensorboard
CONFIG_DIR=/home/poyu39/github/poyu39/emotion-conformer/model/wav2vec2-base
CONFIG_NAME=wav2vec2_base_iemocap

# ========= 執行 Fairseq 預訓練 ========= #
fairseq-hydra-train \
  task.data=$DATA_DIR \
  checkpoint.restore_file=$CHECKPOINT_PATH \
  checkpoint.reset_dataloader=true \
  checkpoint.reset_optimizer=true \
  checkpoint.save_dir=$SAVE_DIR \
  common.tensorboard_logdir=$TENSORBOARD_DIR \
  --config-dir $CONFIG_DIR \
  --config-name $CONFIG_NAME
