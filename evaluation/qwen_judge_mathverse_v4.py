# -*- coding: utf-8 -*-
import os
import re
os.environ['OMP_NUM_THREADS'] = '8'
import json
import tqdm
import glob
import math
import asyncio
import argparse
from concurrent.futures import ProcessPoolExecutor


def extraction_examples():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: <answer>(-2, 1)</answer>
""" # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: <answer>D</answer>
""" # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: <answer>Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)</answer>
""" # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: <answer>null</answer>
""" # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: <answer>22.3</answer>
""" # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: <answer>f(x) = -x^2 - 2x + 1</answer>
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
"""

system_message = (
    "You are a helpful assistant. Your task is to judge if the **Response** is correct based on the **Ground Truth Answer**. "
    "Reply 'true' if the **Response** is correct, otherwise 'false'. Do not need to explain, just 'true' or 'false'."
)

choice_template = (
    "**Choices**: {choices}\n"
    "**Response**: {response}\n"
    "**Ground Truth Answer**: {gt_answer}\n"
)

openend_template = (
    "**Response**: {response}\n"
    "**Ground Truth Answer**: {gt_answer}\n"
)


def parse_args():
    parser = argparse.ArgumentParser(description="MathVerse Qwen Judge ArgParser")
    parser.add_argument("--model_name_or_path", "--model-name-or-path", type=str, default="checkpoints/Qwen2.5-72B-Instruct")
    parser.add_argument("--num_engines", "--num-engines", type=int, default=1)
    parser.add_argument("--eval_basedir", type=str, default="")
    # parser.add_argument("--tensor_parallel_size", type=int, default=8)
    # parser.add_argument("--output_file", type=str, default="output.json", help="Judge results to be written in.")
    return parser.parse_args()


def construct_messages_for_extraction(all_data: list):
    extraction_data_indices = []
    messages = []
    for idx, dataitem in enumerate(all_data):
        response = dataitem['response']
        answer_catcher = re.compile(r"<answer>(.+?)</answer>", flags=re.MULTILINE | re.DOTALL)
        answer_match = re.search(answer_catcher, response)
        if answer_match is not None:
            dataitem['extracted_answer'] = answer_match.group(0)
            continue

        sys_msg = task_description + "\n" + "\n\n".join(extraction_examples())
        user_query = "Model response: " + response + "\n\n" + "Extracted answer: "
        messages.append([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_query},
        ])
        extraction_data_indices.append(idx)
    return messages, extraction_data_indices


def construct_messages_for_judge(dataitem: dict):
    # if dataitem['question_type'] == "multi-choice":
    #     user_query = openend_template.format(response=dataitem['response'], gt_answer=dataitem['answer'])
    # else:   # multi-choice problem
    # answer_catcher = re.compile(r"<answer>(.+?)</answer>", flags=re.MULTILINE | re.DOTALL)
    # answer_match = re.search(answer_catcher, dataitem['response'])
    # response = answer_match.group(1) if answer_match is not None else dataitem['response']
    extracted_answer = dataitem['extracted_answer']
    answer_catcher = re.compile(r"<answer>(.+?)</answer>", flags=re.MULTILINE | re.DOTALL)
    answer_match = re.search(answer_catcher, extracted_answer)
    response = answer_match.group(1) if answer_match is not None else dataitem['response']
    if dataitem['question_type'] == "multi-choice":
        choices_catcher = re.compile(r"(?:Choices|Choice):(.+)", flags=re.MULTILINE | re.DOTALL)
        choices_str = re.search(choices_catcher, dataitem['question_for_eval']).group(1)
        user_query = choice_template.format(
            response=response,
            choices=choices_str,
            gt_answer=dataitem['answer'],
        )

    else:   # free-form
        user_query = openend_template.format(
            response=response,
            gt_answer=dataitem['answer'],
        )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query},
    ]
    return messages


def read_mathverse_eval_data(filepaths: list[str]):
    all_data = []
    for filepath in tqdm.tqdm(filepaths, desc="Loading Data files"):
        data = json.load(open(filepath, "r", encoding="utf-8"))
        for item in data:
            data_id = item.pop('sample_index')
            all_data.append({
                "id": filepath + "###" + data_id,
                # "messages": construct_messages(item),
                **item,
            })
    return all_data


def llm_engine_generate(args, engine_idx: int, all_data: list):
    num_gpus = 8
    gpus_per_engine = num_gpus // args.num_engines
    cuda_visible_devices = ",".join([str(i) for i in range(engine_idx * gpus_per_engine, (engine_idx + 1) * gpus_per_engine)])
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    print(f"Initailizing vLLM on GPU:{cuda_visible_devices}.")

    from vllm import SamplingParams, LLM
    from transformers import Qwen2Tokenizer

    extraction_messages, extraction_indices = construct_messages_for_extraction(all_data)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_name_or_path)
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        stop=[
            tokenizer.eos_token,
            "<|endoftext|>",
            "<|eot_id|>",
        ],
        max_tokens=128,
    )

    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=gpus_per_engine,
        gpu_memory_utilization=0.85,
        # dtype="bfloat16",
        max_num_seqs=2048,
    )

    if len(extraction_messages) > 0:
        extraction_outputs = llm.chat(
            messages=extraction_messages,
            sampling_params=sampling_params,
        )
        for index, output in zip(extraction_indices, extraction_outputs):
            all_data[index]['extracted_answer'] = output.outputs[0].text

    judge_messages = [construct_messages_for_judge(item) for item in all_data]

    outputs = llm.chat(
        messages=judge_messages,
        sampling_params=sampling_params,
    )
    return outputs, all_data


async def llm_generate(args, all_data: list):
    num_engines = args.num_engines
    num_data_per_engine = math.ceil(len(all_data) // num_engines)
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=args.num_engines) as pool:
        tasks = []
        for engine_idx in range(args.num_engines):
            generation_task = loop.run_in_executor(
                pool, llm_engine_generate,
                args, engine_idx,
                all_data[engine_idx * num_data_per_engine:(engine_idx + 1) * num_data_per_engine],
            )
            tasks.append(generation_task)
        results = await asyncio.gather(*tasks)

    outputs = []
    raw_data = []
    for resp_list, raw_data_list in results:
        outputs += resp_list
        raw_data += raw_data_list
    return outputs, raw_data


def main(args):
    eval_results_basedir = args.eval_basedir
    dataset_name = eval_results_basedir.split('/')[1]
    # filepaths = [
    #     os.path.join(eval_results_basedir, dirname, f"{dataset_name}-results.json")
    #     for dirname in os.listdir(eval_results_basedir)
    # ]
    filepaths = []
    for root, dirs, files in os.walk(eval_results_basedir):
        for filename in files:
            if filename.endswith("-results.json"):
                filepaths.append(os.path.join(root, filename))

    def _filter_func(path):
        dirname = os.path.dirname(path)
        if len(glob.glob(os.path.join(dirname, "*qwen_judge_results_v4.json"))) > 0:
            return False
        return True

    filepaths = list(filter(_filter_func, filepaths))
    if len(filepaths) == 0:
        print("All checkpoints have been evaluated.")
        exit(0)

    data_for_judge = read_mathverse_eval_data(filepaths=filepaths)
    # print(data_for_judge[0])

    outputs, data_for_judge = asyncio.run(llm_generate(args, data_for_judge))

    results_to_files = {}
    # judge_catcher = re.compile(r"<judge>(.+?)</judge>", flags=re.DOTALL)
    for idx, output in enumerate(outputs):
        response = output.outputs[0].text
        raw_data = data_for_judge[idx]
        try:
            source_filepath, pid = raw_data['id'].split('###')
        except:
            continue
        # save judge results in the same directory as raw file
        dataset_name = source_filepath.split('/')[1].split('-')[0]
        output_filepath = os.path.join(os.path.dirname(source_filepath), f"{dataset_name}-qwen_judge_results_v4.json")
        results_to_files[output_filepath] = results_to_files.get(output_filepath, []) + [{"id": pid, "extracted_answer": raw_data['extracted_answer'], "judge_result": response}]

    for filepath, results in results_to_files.items():
        json.dump(results, open(filepath, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)
