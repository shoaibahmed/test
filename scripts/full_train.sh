#!/bin/bash

export TOKENIZERS_PARALLELISM=false  # disable tokenizer warning

# Get the DDP args
HEAD_NODE_IP=$1
NUM_NODES=$2
NUM_GPUS_PER_NODE=8
echo "Head node IP: ${HEAD_NODE_IP} / # nodes: ${NUM_NODES} / # GPUs per node: ${NUM_GPUS_PER_NODE}"
RUNNER="torchrun"

# if [ -z "${HEAD_NODE_IP}" ]; then  # Check if HEAD_NODE_IP is given
if [ "${RUNNER}" = "torchrun" ]; then
    if [ "${NUM_NODES}" = "1" ]; then
        TORCH_RUN_ARGS="--standalone --nnodes=1 --nproc_per_node=${NUM_GPUS_PER_NODE}"
    else
        TORCH_RUN_ARGS="--rdzv_id 10344 --rdzv_backend c10d --rdzv_endpoint ${HEAD_NODE_IP}:29500 --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS_PER_NODE}"
    fi
    RUNNER_CMD="torchrun ${TORCH_RUN_ARGS}"
else
    export WORLD_SIZE=${SLURM_NTASKS}
    export RANK=${SLURM_PROCID}
    export LOCAL_RANK=${SLURM_LOCALID}
    export MASTER_ADDR=${HEAD_NODE_IP}
    export MASTER_PORT=29500
    echo "python args / world size: ${WORLD_SIZE} / rank: ${RANK} / local rank: ${LOCAL_RANK} / master addr: ${MASTER_ADDR} / master port: ${MASTER_PORT}"
    RUNNER_CMD="python"
fi

DEFAULT_MODEL="mistral"
MODEL=${3:-$DEFAULT_MODEL}
echo "Using model: ${MODEL}"

EXTRA_ARGS=""

DEFAULT_USE_PRETRAINED="false"
USE_PRETRAINED=${4:-$DEFAULT_USE_PRETRAINED}
echo "Use pretrained model: ${USE_PRETRAINED}"
if [ "${USE_PRETRAINED}" = "true" ]; then
    # Gradient checkpointing doesn't work with FSDP
    EXTRA_ARGS="${EXTRA_ARGS} --use-pretrained-model"
fi

DEFAULT_TRAIN_JUST_HEAD="false"
TRAIN_JUST_HEAD=${5:-$DEFAULT_TRAIN_JUST_HEAD}
echo "Train just head: ${TRAIN_JUST_HEAD}"
if [ "${TRAIN_JUST_HEAD}" = "true" ]; then
    # Gradient checkpointing and model compilation doesn't work with FSDP
    EXTRA_ARGS="${EXTRA_ARGS} --train-only-head --use-ddp --use-gradient-checkpointing"
else
    EXTRA_ARGS="${EXTRA_ARGS} --compile-model"
fi
echo "Extra args: ${EXTRA_ARGS}"

${RUNNER_CMD} multiscale_trainer.py \
    --dataset "fineweb_edu" \
    --dataset-base-dir "./datasets/" \
    --checkpoint-base-dir "./checkpoints/" \
    --preprocessed-dataset-path "" \
    --logs-base-dir "./logs/" \
    --model-name ${MODEL} \
    --model-size 7b \
    --batch-size 1 \
    --sequence-length 2048 \
    --subsample-size -1 \
    --train-epochs 1 \
    --gradient-accumulation-steps 1 \
    --learning-rate 1e-4 \
    --lr-warmup-steps 100 \
    --lr-scheduler "cosine" \
    --min-learning-rate 1e-5 \
    --clip-grad-norm 1.0 \
    --weight-decay 0.1 \
    --eval-after-steps -1 \
    --checkpoint-after-steps -1 \
    --num-workers 16 \
    --dist-socket-timeout 24 \
    --prediction-heads "1,10,100,1000" \
    --prediction-head-weights "1,0.33,0.33,0.33" \
    --multihead-token-weighting-scheme "uniform" \
    --wandb-project "llm-multiscale-pred" \
    --use-harness-evals \
    ${EXTRA_ARGS}
