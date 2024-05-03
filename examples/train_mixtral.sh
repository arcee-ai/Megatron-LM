#!/bin/bash

# Runs Mixtral 8x7B model on 4 GPUs

export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_P2P_DISABLE=1 

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/workspace/checkpoints/Mixtral-8x7B-Instruct-v0.1/megatron/
#TOKENIZER_MODEL=$2
TOKENIZER_MODEL=/workspace/megatron/mixtral_tokenizer/tokenizer.model
#DATA_PATH=$3
# DATA_PATH=/workspace/megatron/test_ds-mixtral_text_document

VOCAB_FILE=/workspace/megatron/mixtral_tokenizer/tokenizer.json
MERGE_FILE=./gpt2-merges.txt
DATA_PATH=test_ds_gpt_text_document

SAVE_PATH=checkpoints/megatron_mixtral_v0.0

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 2048
    --max-position-embeddings 32768
    #--num-layers 32
    --num-layers 1
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
)

#--moe-grouped-gemm
MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 2
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2    
    --moe-grouped-gemm
)

# DATA_ARGS=(
#    --tokenizer-type Llama2Tokenizer
#     --data-path $DATA_PATH
#     --split 99990,8,2
#     --vocab-file $VOCAB_FILE
#     --merge-file $MERGE_FILE
# )

DATA_ARGS="
    --tokenizer-type Llama2Tokenizer    
    --vocab-file $VOCAB_FILE
    --tokenizer-model ${TOKENIZER_MODEL}    
    --data-path $DATA_PATH    
    --split 949,50,1
    --mock-data
"

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 16
    --lr 1e-4
    --train-iters 50
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 100 \
    --save $SAVE_PATH \
    #--load $CHECKPOINT_PATH \
    #--tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"} 
    )
fi

echo "Start training ..."

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}