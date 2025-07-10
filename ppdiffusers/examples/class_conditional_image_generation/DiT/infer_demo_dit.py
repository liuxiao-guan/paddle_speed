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
import os 
os.environ["FLAGS_enable_use_fa"] = "1"
from annotated_types import T
import paddle
from paddlenlp.trainer import set_seed

from ppdiffusers import DDIMScheduler, DiTPipeline
import time 
# from cleanfid import fid

dtype = paddle.float16
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", paddle_dtype=dtype)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

words = ["golden retriever"]  # class_ids [207]
class_ids = pipe.get_label_ids(words)

set_seed(42)
generator = paddle.Generator().manual_seed(0)
start =time.time()
image = pipe(class_labels=class_ids, num_inference_steps=50, generator=generator,guidance_scale=1.5).images[0]
end =time.time()
print(f"time taken: {end-start}")

image.save("result_DiT_golden_retriever.png")

start =time.time()
image = pipe(class_labels=class_ids, num_inference_steps=50, generator=generator,guidance_scale=1.5).images[0]
end =time.time()
print(f"time taken: {end-start}")
image.save("result_DiT_golden_retriever.png")


