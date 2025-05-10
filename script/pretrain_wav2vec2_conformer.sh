#!/bin/bash

# ========= 設定 ========= #
DATA_DIR=/home/poyu39/github/poyu39/emotion-conformer/dataset/LibriSpeech/manifest
SAVE_DIR=/home/poyu39/github/poyu39/emotion-conformer/model/wav2vec2-conformer-base/checkpoints
TENSORBOARD_DIR=/home/poyu39/github/poyu39/emotion-conformer/model/wav2vec2-conformer-base/tensorboard
CONFIG_DIR=/home/poyu39/github/poyu39/emotion-conformer/model/wav2vec2-conformer-base/
CONFIG_NAME=wav2vec2_conformer_base_librispeech

# ========= 執行 Fairseq 預訓練 ========= #
fairseq-hydra-train \
  task.data=$DATA_DIR \
  checkpoint.save_dir=$SAVE_DIR \
  common.tensorboard_logdir=$TENSORBOARD_DIR \
  --config-dir $CONFIG_DIR \
  --config-name $CONFIG_NAME
