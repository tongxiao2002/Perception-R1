# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import defaultdict
from typing import Callable, Dict, List, Tuple, TypedDict

import ray
import torch
import asyncio
from transformers import PreTrainedTokenizer
from concurrent.futures import ProcessPoolExecutor

from ...protocol import DataProto
from ...utils.reward_score import (
    math_compute_score,
    r1v_compute_score,
    boxed_math_verify_compute_score,
    perception_boxed_math_verify_compute_score,
)
from ...utils.reward_score.math_with_visual import (
    batch_visual_key_info_reward,
)
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: RewardConfig):
        self.config = config
        self.tokenizer = tokenizer
        if config.score_function == "math":
            self.compute_score: Callable[[str, str], RewardScore] = math_compute_score
        elif config.score_function == "r1v":
            self.compute_score: Callable[[str, str], RewardScore] = r1v_compute_score
        elif config.score_function == "boxed_math_verify":
            self.compute_score: Callable[[str, str], RewardScore] = boxed_math_verify_compute_score
        elif config.score_function == "perception_boxed_math_verify":
            self.compute_score: Callable[[str, str], RewardScore] = perception_boxed_math_verify_compute_score
        else:
            raise NotImplementedError(f"Unknown score function {config.score_function}.")

    def compute_accuracy_format_reward(
        self,
        reward_tensor, reward_metrics,
        response_strs: list[str],
        valid_response_lengths: list[int],
        ground_truths: list[str],
    ):
        for idx, (response_str, valid_response_length, ground_truth) in \
            enumerate(zip(response_strs, valid_response_lengths, ground_truths)):

            score = self.compute_score(response_str, ground_truth)
            reward_tensor[idx, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

            if self.config.use_ngram_penalty:
                ngram_penalty = self.ngram_penalty_reward(response_str)
                reward_tensor[idx, valid_response_length - 1] += ngram_penalty
                reward_metrics['ngram_penalty'].append(ngram_penalty)

        return reward_tensor, reward_metrics

    async def async_call(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)

        response_ids = data.batch['responses']
        response_mask = data.batch['response_mask']
        ground_truths = data.non_tensor_batch['ground_truth']

        valid_response_lengths = [mask.sum() for mask in response_mask]
        valid_response_ids = [
            ids[:length] for ids, length in zip(response_ids, valid_response_lengths)
        ]
        response_strs = self.tokenizer.batch_decode(
            valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
        )

        if self.config.use_visual_reward:
            visual_key_info = data.non_tensor_batch.get("visual_key_info", [None] * len(response_strs))
            # image_path = data.non_tensor_batch.get("image_path", [None] * len(response_strs))
            visual_reward_task = asyncio.create_task(
                batch_visual_key_info_reward(
                    batch_predict_str=response_strs,
                    batch_visual_key_info=visual_key_info,
                    # image_path=image_path,
                )
            )

        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=1) as pool:
            acc_format_reward_task = loop.run_in_executor(
                pool, self.compute_accuracy_format_reward,
                reward_tensor, reward_metrics,
                response_strs, valid_response_lengths, ground_truths,
            )
            reward_tensor, reward_metrics = await acc_format_reward_task

        if self.config.use_visual_reward:
            visual_rewards, visual_rewards_for_metrics = await visual_reward_task
            reward_metrics['visual'] = visual_rewards_for_metrics
            for idx in range(len(visual_rewards)):
                visual_reward = visual_rewards[idx]
                valid_response_length = valid_response_lengths[idx]
                reward_tensor[idx, valid_response_length - 1] += self.config.visual_reward_factor * visual_reward
                reward_metrics['overall'][idx] += self.config.visual_reward_factor * visual_reward

        return reward_tensor, reward_metrics

    def ngram_penalty_reward(self, response_str: str):
        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)])

        if response_str.strip() == "" or len(response_str.split()) < self.config.ngram_size:
            return 0

        ngrams = set()
        total = 0
        for ng in zipngram(response_str, self.config.ngram_size):
            ngrams.add(ng)
            total += 1

        scaling = 1 - len(ngrams) / total
        reward = scaling * self.config.ngram_max_penalty
        return reward

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor, reward_metrics = asyncio.run(self.async_call(data))
        return reward_tensor, reward_metrics
