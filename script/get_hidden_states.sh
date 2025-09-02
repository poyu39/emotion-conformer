#!/bin/bash

FRONTEND_MODEL_PATH="/home/poyu39/github/poyu39/emotion-conformer/model/wav2vec2-base/librispeech_iemocap/checkpoints/checkpoint_best.pt"
CHECKPOINT_PATH="/home/poyu39/github/poyu39/emotion-conformer/outputs/epoch_100/w2v2_base_ft/checkpoint/fold_1-best_model.pt"
FEATURE_PATH="/home/poyu39/github/poyu39/emotion-conformer/dataset/IEMOCAP_full_release/feature.npy"
LABEL_MAP_PATH="/home/poyu39/github/poyu39/emotion-conformer/dataset/IEMOCAP_full_release/label_map.npy"
EXPORT_HIDDEN_STATES="True"
HIDDEN_DIM="384"

uv run ./src/inference.py \
  --frontend_model_path $FRONTEND_MODEL_PATH \
  --checkpoint_path $CHECKPOINT_PATH \
  --feature_path $FEATURE_PATH \
  --label_map_path $LABEL_MAP_PATH \
  --export_hidden_states $EXPORT_HIDDEN_STATES \
  --hidden_dim $HIDDEN_DIM \

