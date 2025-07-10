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
# import os 
# os.environ["FLAGS_enable_use_fa"] = "1"
# from annotated_types import T
# import torch

# from diffusers import DDIMScheduler, DiTPipeline
# import time 
# # from cleanfid import fid

# dtype = torch.float32
# pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=dtype)
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")
# words = ["golden retriever"]  # class_ids [207]
# class_ids = pipe.get_label_ids(words)


# generator = torch.Generator().manual_seed(24)
# start =time.time()
# image = pipe(class_labels=class_ids, num_inference_steps=50, generator=generator).images[0]
# end =time.time()
# print(f"time taken: {end-start}")

# image.save("result_DiT_golden_retriever.png")

# start =time.time()
# image = pipe(class_labels=class_ids, num_inference_steps=50, generator=generator).images[0]
# end =time.time()
# print(f"time taken: {end-start}")
# image.save("result_DiT_golden_retriever.png")




from diffusers import DiTPipeline, DPMSolverMultistepScheduler,DDIMScheduler
import torch
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
import time
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
print(pipe.transformer.transformer_blocks[0].attn1.processor)  
# pipe.enable_xformers_memory_efficient_attention()
 # pick words from Imagenet class labels
pipe.labels  # to print all available words
#pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
words = ["white shark"]

class_ids = pipe.get_label_ids(words)

generator = torch.manual_seed(33)
start =time.time()
output = pipe(class_labels=class_ids, num_inference_steps=50, generator=generator)
end =time.time()
print(f"time taken: {end-start}")
image = output.images[0]  # label 'white shark'

start =time.time()
output = pipe(class_labels=class_ids, num_inference_steps=50, generator=generator)
end =time.time()
print(f"time taken: {end-start}")
image = output.images[0] 
