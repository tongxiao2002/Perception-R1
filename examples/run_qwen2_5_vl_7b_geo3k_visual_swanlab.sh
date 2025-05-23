set -x
ulimit -u 32768
ulimit -n 32768

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

export VLLM_SERVER_BASE_URL="YOUR VLLM SERVER URL"
export VLLM_SERVER_API_KEY="YOUR VLLM SERVER API KEY"
export VLLM_MODEL_NAME="Qwen/Qwen2.5-32B-Instruct-AWQ"


export EXPERIMENT_NAME="qwen2_5_vl_7b_geo3k_with_visual_v2_factor_0.7_with_ngram_penalty"
export SWANLAB_MODE="local"
export SWANLAB_DIR="swanlogs/${EXPERIMENT_NAME}"

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags, and the answer process MUST BE enclosed within <answer> </answer> tags.
 The final answer MUST BE put in \boxed{} in <answer> </answer> tags."""

# ray start --head --num-cpus=64
micro_batch_size_per_device_for_update=8
micro_batch_size_per_device_for_experience=16
gradient_accumulation_steps=2
nnodes=1
n_gpus_per_node=8
global_batch_size=$((${n_gpus_per_node} * ${nnodes} * ${micro_batch_size_per_device_for_update} * ${gradient_accumulation_steps}))
rollout_batch_size=$((${global_batch_size} * 1))

# RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --runtime-env=examples/multinode_ray_runtime_env.yaml --no-wait -- \
python3 -m verl.trainer.main \
    config=examples/default_config.yaml \
    data.train_files=data/geo3k_with_visual_key_info.parquet@train \
    data.val_files=data/varsitytutor-data-test.parquet@train \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.rollout_batch_size=${rollout_batch_size} \
    data.min_pixels=3136 \
    data.max_pixels=1806336 \
    algorithm.use_kl_loss=false \
    algorithm.disable_kl=true \
    worker.reward.use_visual_reward=true \
    worker.reward.visual_reward_factor=0.7 \
    worker.reward.use_ngram_penalty=true \
    worker.reward.ngram_size=20 \
    worker.reward.ngram_max_penalty=-2.0 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${global_batch_size} \
    worker.actor.micro_batch_size_per_device_for_update=${micro_batch_size_per_device_for_update} \
    worker.actor.micro_batch_size_per_device_for_experience=${micro_batch_size_per_device_for_experience} \
    worker.actor.optim.lr=1e-6 \
    worker.actor.optim.weight_decay=1e-3 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.offload.offload_params=false \
    worker.ref.offload.offload_params=false \
    worker.rollout.n=8 \
    worker.rollout.enforce_eager=false \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.tensor_parallel_size=4 \
    worker.reward.score_function=boxed_math_verify \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.total_episodes=25 \
    trainer.val_freq=5 \
    trainer.save_freq=10 \
    trainer.nnodes=${nnodes}
