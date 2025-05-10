#!/bin/bash

# ==== 使用者參數 ====

# 原始 LibriSpeech 解壓縮資料夾
DATASET_DIR=/home/poyu39/github/poyu39/emotion-conformer/dataset/LibriSpeech

# 要產生 manifest 的目的地
DEST_DIR=/home/poyu39/github/poyu39/emotion-conformer/dataset/_librispeech

mkdir -p $DEST_DIR

SRC_DIR=/home/poyu39/github/poyu39/emotion-conformer/src/

python $SRC_DIR/gen_manifest.py \
  train-clean-100 train-clean-360 train-other-500 \
  --root $DATASET_DIR \
  --output-name train \
  --dest $DEST_DIR \

python $SRC_DIR/gen_manifest.py \
  dev-clean dev-other \
  --root $DATASET_DIR \
  --output-name valid \
  --dest $DEST_DIR \

python $SRC_DIR/gen_manifest.py \
    test-clean test-other \
    --root $DATASET_DIR \
    --output-name test \
    --dest $DEST_DIR \

echo "Done! Manifest files are generated in $DEST_DIR"