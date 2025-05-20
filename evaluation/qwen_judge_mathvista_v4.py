import os
import re
import json
import tqdm
import glob
import argparse
from transformers import Qwen2Tokenizer
from vllm import SamplingParams, LLM


def extraction_examples():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: <answer>14</answer>
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: <answer>0.6</answer>
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: <answer>1.45</answer>
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: <answer>[2007, 2008]</answer>
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: <answer>B</answer>
"""

    return [example_1, example_2, example_3, example_4, example_5]


task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
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
    parser = argparse.ArgumentParser(description="MathVista Qwen Judge ArgParser")
    parser.add_argument("--model_name_or_path", "--model-name-or-path", type=str, default="checkpoints/Qwen2.5-72B-Instruct")
    parser.add_argument("--eval_basedir", "--eval-basedir", type=str, default="")
    parser.add_argument("--tensor_parallel_size", "--tensor-parallel-size", type=int, default=8)
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

        sys_msg = task_description + "\n" + "\n".join(extraction_examples())
        user_query = dataitem['query'] + "\n\n" + "Model response: " + response + "\n\n" + "Extracted answer: "
        messages.append([
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_query},
        ])
        extraction_data_indices.append(idx)
    return messages, extraction_data_indices


def construct_messages_for_judge(dataitem: dict):
    def make_choice_string(choices: list):
        choice_string = []
        for idx, choice in enumerate(choices):
            choice_string.append(
                f"{chr(ord('A') + idx)}. {choice}"
            )
        return "\n".join(choice_string)

    extracted_answer = dataitem['extracted_answer']
    answer_catcher = re.compile(r"<answer>(.+?)</answer>", flags=re.MULTILINE | re.DOTALL)
    answer_match = re.search(answer_catcher, extracted_answer)
    response = answer_match.group(1) if answer_match is not None else dataitem['response']
    if 'choices' not in dataitem or dataitem['choices'] is None:
        user_query = openend_template.format(response=response, gt_answer=dataitem['answer'])
    else:   # multi-choice problem
        user_query = choice_template.format(
            response=response,
            choices=make_choice_string(dataitem['choices']),
            gt_answer=dataitem['answer'],
        )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query},
    ]
    return messages


def read_mathvista_eval_data(filepaths: list[str]):
    all_data = []
    for filepath in tqdm.tqdm(filepaths, desc="Loading Data files"):
        data = json.load(open(filepath, "r", encoding="utf-8"))
        for key, item in data.items():
            data_id = item.pop('pid') if 'pid' in item else item.pop('id')
            all_data.append({
                "id": filepath + "###" + data_id,
                # "messages": construct_messages(item),
                **item,
            })
    return all_data


def main(args):
    model_name_or_path = args.model_name_or_path
    # filepaths = args.files
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

    data_for_judge = read_mathvista_eval_data(filepaths=filepaths)
    # print(data_for_judge[0])

    extraction_messages, extraction_indices = construct_messages_for_extraction(data_for_judge)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path)

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
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        dtype="bfloat16",
        max_num_seqs=1024,
    )

    if len(extraction_messages) > 0:
        extraction_outputs = llm.chat(
            messages=extraction_messages,
            sampling_params=sampling_params,
        )
        for index, output in zip(extraction_indices, extraction_outputs):
            data_for_judge[index]['extracted_answer'] = output.outputs[0].text

    judge_messages = [construct_messages_for_judge(item) for item in data_for_judge]

    outputs = llm.chat(
        messages=judge_messages,
        sampling_params=sampling_params,
    )

    results_to_files = {}
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
