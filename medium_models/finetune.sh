#!/bin/bash


TASK=${TASK:-SST-2}
K=${K:-16}
SEED=${SEED:-42}
BS=${BS:-8}
LR=${LR:-1e-5}
STEP=${STEP:-1000}
EVAL_STEP=${EVAL_STEP:-100}
MODEL=${MODEL:-"../../../Model/roberta-large"}

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

echo "TASK: $TASK"
echo "K: $K"
echo "Seed: $SEED"
echo "BS: $BS"
echo "LR: $LR"
echo "Step: $STEP; Eval step: $EVAL_STEP"

GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
EXTRA_TAG=${EXTRA_TAG:-ft}
TAG=${TAG:-k${K}-${MODELNAME}-${EXTRA_TAG}}

echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

# 创建日志目录
mkdir -p log_dir
# Redirect all output to a log file based on the TAG
exec &> >(tee "log_dir/${TASK}-${GR_TAG}-${TAG}.log")

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    bash run_fewshot.sh --per_device_train_batch_size $BS --learning_rate $LR --eval_steps $EVAL_STEP \
    $@
