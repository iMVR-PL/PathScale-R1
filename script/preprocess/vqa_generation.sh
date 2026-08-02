#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/.env"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

cd "${ROOT}/preprocess/generate_vqa_data"

# Extract visual features
python extract_visual_features.py \
  --input-jsonl $PROCESSED_DIR/feature_extracted/expert_captions.jsonl \
  --output-jsonl $PROCESSED_DIR/feature_extracted/visual_features.jsonl \
  --model gpt-4.1-2025-04-14

# VQA Sampling (assign question type (fixed or free) to each sample)
python vqa_sampling.py \
  --input-jsonl $PROCESSED_DIR/feature_extracted/visual_features.jsonl \
  --output-json $PROCESSED_DIR/cross_vqa_generated/vqa_assignments.json \
  --seed 42

# VQA Generation based on revised prompt after text-only adversarial evaluation
python vqa_generation.py \
  --input-jsonl $PROCESSED_DIR/feature_extracted/visual_features.jsonl \
  --output-jsonl $PROCESSED_DIR/cross_vqa_generated/vqa_grouped.jsonl \
  --assignment-json $PROCESSED_DIR/cross_vqa_generated/vqa_assignments.json \
  --types A,B,C,D,E \
  --split-by-type \
  --prompt-dir $ROOT/preprocess/prompts \
  --model gpt-5.2-2025-12-11

# Merge generated VQA
python merge_mcq_types.py \
  --input-dir $PROCESSED_DIR/cross_vqa_generated \
  --output $PROCESSED_DIR/cross_vqa_merged_balanced/vqa_grouped_merged.jsonl

# Balance MCQ options
python balance_mcq_answers.py \
  --input-jsonl $PROCESSED_DIR/cross_vqa_merged_balanced/vqa_grouped_merged.jsonl \
  --output-jsonl $PROCESSED_DIR/cross_vqa_merged_balanced/vqa_grouped_merged_balanced.jsonl \
  --stats-json $PROCESSED_DIR/cross_vqa_merged_balanced/vqa_grouped_merged_balanced_stats.json \
  --seed 42

# Train-val-test split
## Extract wsi and paths 
python extract_wsi_paths.py \
  --input $PROCESSED_DIR/feature_extracted/triplet_merged.jsonl \
  --output $PROCESSED_DIR/train_test_split/triplet_merged_paths.jsonl

## Generate WSI-based split
python wsi_split.py \
  --input $PROCESSED_DIR/train_test_split/triplet_merged_paths.jsonl \
  --output $PROCESSED_DIR/train_test_split/wsi_train_test_split.csv \
  --seed 42

## Finalize MCQ data with train_test_split info
python finalize_mcq.py \
  --vqa-jsonl $PROCESSED_DIR/cross_vqa_merged_balanced/vqa_grouped_merged_balanced.jsonl \
  --split-csv $PROCESSED_DIR/train_test_split/wsi_split.csv \
  --output $PROCESSED_DIR/cross_vqa_finalized/all_vqa.json \
  --data-root $DATA_DIR
