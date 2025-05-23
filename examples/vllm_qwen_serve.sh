#!/bin/bash
set -x

export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_LOGGING_LEVEL=ERROR
export OMP_NUM_THREADS=8


vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
    --port 16657 \
    --tensor-parallel-size 8 \
    --api-key sk-abc123 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching
