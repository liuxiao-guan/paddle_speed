# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse

import paddle


from paddlemix.models.qwen2_5_vl import MIXQwen2_5_Tokenizer
from paddlemix.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from paddlemix.processors.qwen2_5_vl_processing import (
    Qwen2_5_VLImageProcessor,
    Qwen2_5_VLProcessor,
    process_vision_info,
)
from paddlemix.utils.log import logger
from paddle.distributed import fleet



def main(args):
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
                    "dp_degree": 1,
                    "mp_degree": args.mp_degree,
                    "pp_degree": 1,
                    "sharding_degree": 1,
                }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    tensor_parallel_rank = hcg.get_model_parallel_rank()
    paddle.seed(seed=0)
    compute_dtype = args.dtype
    if "npu" in paddle.get_device():
        is_bfloat16_supported = True
    else:
        is_bfloat16_supported = paddle.amp.is_bfloat16_supported()
    if compute_dtype == "bfloat16" and not is_bfloat16_supported:
        logger.warning("bfloat16 is not supported on your device,change to float32")
        compute_dtype = "float32"
    print("compute_dtype", compute_dtype)
    paddle.set_default_dtype(compute_dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, tensor_parallel_degree=args.mp_degree, tensor_parallel_rank=tensor_parallel_rank, dtype=compute_dtype, tensor_parallel_output=False)
    model.eval()
    image_processor = Qwen2_5_VLImageProcessor()
    tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(args.model_path)
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer)

    # min_pixels = 256*28*28 # 200704
    # max_pixels = 1280*28*28 # 1003520
    # processor = Qwen2VLProcessor(image_processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{args.image_file}",
                },
                {"type": "text", "text": f"{args.question}"},
            ],
        }
    ]

    # Preparation for inference
    image_inputs, video_inputs = process_vision_info(messages)

    # question = messages[0]["content"][1]["text"]
    # image_pad_token = "<|vision_start|><|image_pad|><|vision_end|>"
    # text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_pad_token}{question}<|im_end|>\n<|im_start|>assistant\n"
    # text = [text]
    text = [processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pd",
    )

    if args.benchmark:
        import time

        start = 0.0
        total = 0.0
        for i in range(20):
            if i > 10:
                start = time.time()
            with paddle.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p
                )  # already trimmed in paddle
                output_text = processor.batch_decode(
                    generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            if i > 10:
                total += time.time() - start
        print("s/it: ", total / 10)
        print(f"GPU memory_allocated: {paddle.device.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU max_memory_allocated: {paddle.device.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU memory_reserved: {paddle.device.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        print(f"GPU max_memory_reserved: {paddle.device.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")
        print("output_text:\n", output_text)

    else:
        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p
        )  # already trimmed in paddle
        output_text = processor.batch_decode(
            generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(f"GPU memory_allocated: {paddle.device.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU max_memory_allocated: {paddle.device.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU memory_reserved: {paddle.device.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        print(f"GPU max_memory_reserved: {paddle.device.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")
        print("output_text:\n", output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--question", type=str, default="Describe this image.")
    parser.add_argument("--image_file", type=str, default="paddlemix/demo_images/examples_image1.jpg")
    parser.add_argument("--top_p", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--mp_degree", type=int, default=2)
    args = parser.parse_args()
    main(args)
