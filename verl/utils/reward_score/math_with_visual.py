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

import os
import re
import base64
import asyncio
from openai import AsyncOpenAI
# from openai.types.chat import ChatCompletion
from codetiming import Timer

# from mathruler.grader import grade_answer
from math_verify import parse, verify

system_prompt2 = (
    "Given an image for a multimodal math problem, "
    "your task is to evaluate the accuracy of the descriptions about the image in the given 'Response' with respect to the provided image.\n"
    "You should compare these descriptions against the image to judge their accuracy, "
    "and finally give a numerical value between 0 and 1 representing the accuracy, where 1 represents completely correct, 0 represents completely wrong.\n"
    "Note that you should only pay attention to descriptions about the image, and should ignore any content about logical reasoning and computations.\n"
    "Finally you should wrap the numerical value in the <info1></info1> tags, e.g., <info1>0.6</info1>.\n"
)

user_prompt2 = (
    "Response: {response}\n"
)

system_prompt = (
    "Given visual key information for a multimodal math problem, determine whether the 'Response' includes each piece of key information. "
    "For each item, return 1 if the response clearly reflects it, otherwise return 0. "
    "Respond using the format: <info1>1 or 0</info1>, <info2>1 or 0</info2>, etc. "
    "Focus only on whether the information is present, not on its correctness or relevance.\n"
    "Here are some examples:\n"
    "Visual Key Information: <info1>$JKLM$ is a parallelogram.</info1>\n<info2>Length of side $JK$ is given as $3f - 6$.</info2>\n<info3>Length of opposite side $ML$ is given as $2f + 8$.</info3>\n"
    "Response: To find the value of \\( f \\) in the parallelogram, we need to use the properties of a parallelogram. Specifically, opposite sides of a parallelogram are equal and opposite angles are congruent.\n\nGiven:\n- \\( \\angle J \\) is \\( 56^\\circ \\)\n- \\( \\angle M \\) is \\( (3d - 2)^\\circ \\)\n- \\( \\overline{JK} \\) is \\( 3f - 6 \\)\n- \\( \\overline{ML} \\) is \\( 2f + 8 \\)\n\nSince \\( \\overline{JK} \\) and \\( \\overline{ML} \\) are opposite sides of the parallelogram, we have:\n\\[ 3f - 6 = 2f + 8 \\]\n\nNext, we solve for \\( f \\):\n1. Subtract \\( 2f \\) from both sides:\n\\[ 3f - 2f - 6 = 8 \\]\n\\[ f - 6 = 8 \\]\n\n2. Add 6 to both sides:\n\\[ f = 8 + 6 \\]\n\\[ f = 14 \\]\n\nThus, the value of \\( f \\) is \\(\\boxed{14}\\).\n"
    "Judgement: <info1>0</info1>\n<info2>1</info2>\n<info3>1</info3>\n"
    "Visual Key Information: <info1>QRST is a quadrilateral.</info1>\n<info2>The sum of the interior angles is 360 degrees.</info2>\n<info3>$m\\angle Q = (2x + 5)^\\circ$</info3>\n<info4>$m\\angle R = x^\\circ$</info4>\n<info5>$m\\angle S = (2x + 7)^\\circ$</info5>\n<info6>$m\\angle T = x^\\circ$</info6>\n"
    "Response: To find the measure of angle \\( \\angle S \\) in the given rhombus \\( QRST \\), we need to use the properties of a rhombus. Specifically, the sum of the interior angles of a quadrilateral is \\( 360^\\circ \\) and the opposite angles of a rhombus are equal.\n\nGiven:\n- \\( \\angle Q = (2x + 5)^\\circ \\)\n- \\( \\angle R = x^\\circ \\)\n- \\( \\angle T = x^\\circ \\)\n- \\( \\angle S = ? \\)\n\nSince \\( QRST \\) is a rhombus, the opposite angles are equal. Therefore, \\( \\angle Q = \\angle T \\) and \\( \\angle R = \\angle S \\).\n\nWe can set up the equation for the sum of the interior angles:\n\\[\n\\angle Q + \\angle R + \\angle S + \\angle T = 360^\\circ\n\\]\n\nSubstitute the given values:\n\\[\n(2x + 5) + x + \\angle S + x = 360^\\circ\n\\]\n\nSince \\( \\angle R = \\angle S \\), we can simplify the equation:\n\\[\n(2x + 5) + x + x + x = 360^\\circ\n\\]\n\nCombine like terms:\n\\[\n(2x + 5) + 3x = 360^\\circ\n\\]\n\\[\n5x + 5 = 360^\\circ\n\\]\n\nSubtract 5 from both sides:\n\\[\n5x = 355^\\circ\n\\]\n\nDivide by 5:\n\\[\nx = 71^\\circ\n\\]\n\nNow, we can find \\( \\angle S \\):\n\\[\n\\angle S = x = 71^\\circ\n\\]\n\nThus, the measure of \\( \\angle S \\) is:\n\\[\n\\boxed{71^\\circ}\n\\]\n"
    "Judgement: <info1>1</info1>\n<info2>1</info2>\n<info3>1</info3>\n<info4>1</info4>\n<info5>0</info5>\n<info6>1</info6>\n"
    "Visual Key Information: <info1>Measure of angle ABD = $(4y + 4)^\circ$</info1>\n<info2>Measure of angle CDB = $(y^2 - 1)^\circ$</info2>\n<info3>Angle ABD and Angle CDB are alternate interior angles.</info3>\n"
    "Response: Since quadrilateral ABCD is a rectangle, all its interior angles are right angles, meaning they are all 90 degrees. This gives us the following information about the angles:\n\n1. Angle A is labeled as \( (4x + 5)^\circ \).\n2. Angle B is labeled as \( (9x + 20)^\circ \).\n3. Angle C is labeled as \( (4y + 4)^\circ \).\n4. Angle D is labeled as \( (y^2 - 1)^\circ \).\n\nSince angle A is a right angle, we have:\n\[ 4x + 5 = 90 \]\nSolving for \( x \):\n\[ 4x = 85 \]\n\[ x = \frac{85}{4} = 21.25 \]\n"
    "Judgement: <info1>0</info1>\n<info2>0</info2>\n<info3>0</info3>\n"
)

user_prompt = (
    "Visual Key Information: {visual_key_info}\n"
    "Response: {response}\n"
)
vllm_server_base_url = os.environ.get("VLLM_SERVER_BASE_URL", "http://127.0.0.1:8000/v1")
vllm_server_api_key = os.environ.get("VLLM_SERVER_API_KEY", "sk-abc123")
vllm_model_name = os.environ.get("VLLM_MODEL_NAME", "Qwen2.5-32B-Instruct")
async_client = AsyncOpenAI(
    base_url=vllm_server_base_url,
    api_key=vllm_server_api_key,
    max_retries=2,
)


def convert_image_to_base64(image_path: str):
    # image = Image.open(image_path)
    with open(image_path, "rb") as img_file:
        encoded_str = base64.b64encode(img_file.read()).decode("utf-8")

    img_format = os.path.splitext(image_path)[1][1:].lower()
    return f"data:image/{img_format};base64,{encoded_str}"


def boxed_math_verify_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"^<think>\n.*?\n</think>\n<answer>\n.*?\\boxed{.+?}.*?\n</answer>$", re.DOTALL)
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


def extract_visual_scores(judge_result: str):
    pattern = r".*<info{idx}>(.+?)</info{idx}>"
    index = 1
    visual_scores = []
    while True:
        ptn = pattern.format(idx=index)
        try:
            match = re.search(ptn, judge_result, flags=re.MULTILINE | re.DOTALL)
        except Exception as e:
            print(f"Error: {e}")
            return visual_scores

        if match is None:
            break
        try:
            score = float(match.group(1))
        except Exception:
            score = 0.0
        score = max(0, min(1, score))
        visual_scores.append(score)
        index += 1
    return visual_scores


async def batch_visual_key_info_reward(batch_predict_str: list[str], batch_visual_key_info: list[str], **kwargs) -> float:
    semaphore = asyncio.Semaphore(512)

    async def _send_request(msg):
        async with semaphore:
            try:
                response = await async_client.chat.completions.create(
                    model=vllm_model_name,
                    messages=msg,
                    temperature=0.0,
                    n=1,
                    max_completion_tokens=512,
                    # extra_body={
                    #     "chat_template_kwargs": {"enable_thinking": False}
                    # },
                )
                return response
            except Exception as e:
                print(f"Error: {e}")
                return None

    rewards = [0] * len(batch_predict_str)
    valid_indexes = []
    tasks = []
    # image_path = kwargs.get("image_path", [None] * len(batch_predict_str))
    for idx, (predict_str, visual_key_info) in enumerate(zip(batch_predict_str, batch_visual_key_info)):
        # if not visual_key_info or not image_path[idx]:
        if not visual_key_info:
            continue

        message = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt.format(visual_key_info=visual_key_info, response=predict_str),
            },
        ]
        # message = [
        #     {
        #         "role": "system",
        #         "content": system_prompt2,
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image_url",
        #                 "image_url": {"url": convert_image_to_base64(image_path[idx])}
        #             },
        #             {"type": "text", "text": user_prompt2.format(response=predict_str)},
        #         ],
        #     },
        # ]
        # messages.append(message)
        valid_indexes.append(idx)
        tasks.append(asyncio.create_task(_send_request(message)))

    with Timer(name=f"{vllm_model_name} Judge Time") as timer:
        # timer.start()
        responses = await asyncio.gather(*tasks)
    print(f"{timer.name}: {timer.last}")
    print(f"{vllm_model_name} Processed {len(tasks)} Requests.")
    responses = [resp.choices[0].message.content for resp in responses]

    # 防止没有 visual_key_info 对 log 中的 visual rewards 产生影响
    rewards_for_metrics = []
    for valid_index, response in zip(valid_indexes, responses):
        if response is None:
            rewards[valid_index] = 0.0
            rewards_for_metrics.append(0.0)
            continue

        visual_scores = extract_visual_scores(response)
        reward = sum(visual_scores) / (len(visual_scores) + 1e-6)

        rewards[valid_index] = reward
        rewards_for_metrics.append(reward)
    rewards_for_metrics = [0] if len(rewards_for_metrics) == 0 else rewards_for_metrics

    return rewards, rewards_for_metrics


def main(predict_str: str, ground_truth: str, **kwargs) -> float:
    # visual_reward = await asyncio.create_task(visual_key_info_reward(predict_str, kwargs['visual_key_info']))
    accuracy_reward = boxed_math_verify_accuracy_reward(predict_str, ground_truth)
    format_reward = boxed_math_verify_format_reward(predict_str)

    return {
        "overall": 1 * accuracy_reward + 0.1 * format_reward,
        "accuracy": accuracy_reward,
        "format": format_reward,
    }


def boxed_math_verify_with_visual_compute_score(predict_str: str, ground_truth: str, **kwargs):
    # return asyncio.run(main(predict_str=predict_str, ground_truth=ground_truth, **kwargs))
    return main(predict_str=predict_str, ground_truth=ground_truth, **kwargs)
