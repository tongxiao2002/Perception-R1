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

import re

# from mathruler.grader import grade_answer
from math_verify import parse, verify


def boxed_math_verify_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"^<think>\n.*?\n</think>\n<answer>\n.*?\\boxed{.+?}.*?\n</answer>$", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def preception_boxed_math_verify_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"^<perception>\n.*?\n</perception>\n<think>\n.*?\n</think>\n<answer>\n.*?\\boxed{.+?}.*?\n</answer>$", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def boxed_math_verify_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        ground_truth = ground_truth.strip()
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if verify(parse(ground_truth), parse(given_answer)):
            return 1.0
    except Exception:
        pass

    return 0.0


def boxed_math_verify_compute_score(predict_str: str, ground_truth: str) -> float:
    accuracy_reward = boxed_math_verify_accuracy_reward(predict_str, ground_truth)
    format_reward = boxed_math_verify_format_reward(predict_str)
    return {
        "overall": 0.9 * accuracy_reward + 0.1 * format_reward,
        "accuracy": accuracy_reward,
        "format": format_reward,
    }


def perception_boxed_math_verify_compute_score(predict_str: str, ground_truth: str) -> float:
    accuracy_reward = boxed_math_verify_accuracy_reward(predict_str, ground_truth)
    format_reward = preception_boxed_math_verify_format_reward(predict_str)
    return {
        "overall": 0.9 * accuracy_reward + 0.1 * format_reward,
        "accuracy": accuracy_reward,
        "format": format_reward,
    }
