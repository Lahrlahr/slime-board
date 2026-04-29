#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

# Detect NVLink
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Common args
CKPT_ARGS=(
   --hf-checkpoint /data/huangguang/model/Qwen/Qwen3-VL-2B-Instruct
   # qwen3 vl model has rotary base 5000000, set it when applicable
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data /data/huangguang/data/chenhegu/geo3k_imgurl/train.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 3000
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 64
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'

EVAL_ARGS=(
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

# Wandb args (only if WANDB_API_KEY is set)
export WANDB_API_KEY=e20876edfa0ada0582f52a7982047c6f852c6969
if [ -n "$WANDB_API_KEY" ]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-project slime
      --wandb-group USB
      --wandb-key ${WANDB_API_KEY}
      --disable-wandb-random-suffix
   )
else
   WANDB_ARGS=()
fi



# Backend-specific args
# megatron backend
BACKEND_ARGS=(
   --train-backend megatron
   --load /data/huangguang/model/Qwen/Qwen3-VL-2B-Instruct
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
)


MODEL_ARGS_ROTARY_BASE=5000000 source /project/slime/scripts/models/qwen3-1.7B.sh


export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 2 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265


# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/project/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"
MISC_ARGS=(
    --rollout-num-gpus 1
    --actor-num-nodes 1
    --actor-num-gpus-per-node 1
    --critic-num-nodes 1
    --critic-num-gpus-per-node 1
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}