import os
import io
import json
import tqdm
import argparse
import base64
# from PIL import Image
from datasets import load_dataset, Image
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLProcessor
from codetiming import Timer
from functools import partial


SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <think> </think> tags, and the answer process MUST BE enclosed within <answer> </answer> tags. "
    "The final answer MUST BE put in \\boxed{} in <answer> </answer> tags."
)


def encode_image_to_base64(img: bytes):
    img_str = base64.b64encode(img).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def evaluate(args):
    dataset = load_dataset(
        "parquet",
        data_files=['MathVerse/testmini.parquet'],
        split='train',
    )
    dataset = dataset.cast_column("image", Image(decode=True))
    inputs = []
    processor = Qwen2_5_VLProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
        max_num_seqs=512,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=1,
        n=1,
        max_tokens=4096,
        skip_special_tokens=False,
    )

    for idx, data_item in tqdm.tqdm(enumerate(dataset)):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": data_item['image'],
                    },
                    {
                        "type": "text",
                        "text": data_item['question'] + ' ' + SYSTEM_PROMPT,
                    },
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data
            },
        })

    with Timer(name="vLLM Perf Timer", text="{name}: Cosume {minutes:.2f} minutes for generation.") as timer:
        model_outputs = llm.generate(inputs, sampling_params=sampling_params)
    print(f"{timer.name}: {timer.last}")

    outputs = []
    for dataitem, output in zip(dataset, model_outputs):
        del dataitem['image']

        outputs.append({
            **dataitem,
            "response": output.outputs[0].text
        })
    output_filepath = os.path.join(args.out_dir, args.filename)
    json.dump(outputs, open(output_filepath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print('Results saved to {}'.format(output_filepath))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--datasets', type=str, default='MathVerse')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='tests')
    parser.add_argument('--filename', type=str, default='mathverse_test.json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    evaluate(args)
