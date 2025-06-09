#!/bin/bash

LIBRISPEECH_DIR=/path/to/librispeech
IEMOCAP_DIR=/path/to/iemocap

PRETRAIN_DIR=/path/to/pretrain
FINETUNE_DIR=/path/to/finetune

SRC_DIR=/path/to/src

mkdir -p $PRETRAIN_DIR
mkdir -p $FINETUNE_DIR

# ===== 生成 LibriSpeech 訓練資料到 PRETRAIN_DIR =====

python $SRC_DIR/make_manifest.py \
  train-clean-100 train-clean-360 train-other-500 \
  --dataset-name librispeech \
  --root $LIBRISPEECH_DIR \
  --output-name train \
  --dest $PRETRAIN_DIR

# ===== 生成 LibriSpeech valid / test 到 PRETRAIN_DIR =====

python $SRC_DIR/make_manifest.py \
  dev-clean dev-other \
  --dataset-name librispeech \
  --root $LIBRISPEECH_DIR \
  --output-name valid \
  --dest $PRETRAIN_DIR

python $SRC_DIR/make_manifest.py \
  test-clean test-other \
  --dataset-name librispeech \
  --root $LIBRISPEECH_DIR \
  --output-name test \
  --dest $PRETRAIN_DIR

# ===== 生成 IEMOCAP 資料到 FINETUNE_DIR =====

python $SRC_DIR/make_manifest.py \
  Session1 Session2 Session3 \
  --dataset-name iemocap \
  --root $IEMOCAP_DIR \
  --ext wav \
  --output-name train \
  --dest $FINETUNE_DIR

python $SRC_DIR/make_manifest.py \
  Session4 \
  --dataset-name iemocap \
  --root $IEMOCAP_DIR \
  --ext wav \
  --output-name valid \
  --dest $FINETUNE_DIR

python $SRC_DIR/make_manifest.py \
  Session5 \
  --dataset-name iemocap \
  --root $IEMOCAP_DIR \
  --ext wav \
  --output-name test \
  --dest $FINETUNE_DIR

echo "Done! LibriSpeech data saved to $PRETRAIN_DIR, IEMOCAP data saved to $FINETUNE_DIR"