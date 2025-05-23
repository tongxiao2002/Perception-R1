# Perception-R1

## Overview

## Getting Started
### Data Preparation
You can download our data from [Google Drive](https://drive.google.com/), and then put it in the `data` directory.

### Training
You should first start a vllm server:
```shell
bash examples/vllm_qwen_serve.sh
```
Then modify the `VLLM_SERVER_BASE_URL`, `VLLM_SERVER_API_KEY` and `VLLM_MODEL_NAME` fields in `examples/run_qwen2_5_vl_7b_geo3k_visual_swanlab.sh` based on your config.

Once the vllm server is initialized, you can run following scripts to start training:
```shell
bash examples/run_qwen2_5_vl_7b_geo3k_visual_swanlab.sh
```

## Modifications
Our modifications for incorporating the visual perception reward are primarily implemented in the following modules:
- `verl/workers/reward/custom.py`: Contains the core logic for assigning rewards to model-generated responses, including the implementation of the N-gram penalty reward.
- `verl/utils/reward_score/math_with_visual.py`: Implements the visual perception reward, which evaluates the consistency between visual annotations and generated responses.
- `verl/utils/reward_score/boxed_math_verify.py`: Implements the accuracy and format rewards used during the training process to ensure correctness and structured output format.

