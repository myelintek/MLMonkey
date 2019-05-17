#!/bin/bash

set -e

export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=4000


DATASET_DIR='/data'

SEED=${1:-"1"}
TARGET=${2:-"24.00"}

# run training
for ((rank=0;rank<WORLD_SIZE-1;rank++)); do
  RANK=$rank python3 train.py \
    --local_rank $rank \
    --dataset-dir ${DATASET_DIR} \
    --seed $SEED \
    --target-bleu $TARGET &

done

RANK=$(($WORLD_SIZE-1)) python3 train.py \
    --local_rank $(($WORLD_SIZE-1)) \
    --dataset-dir ${DATASET_DIR} \
    --seed $SEED \
    --target-bleu $TARGET 
