# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import paddle

from paddlenlp.transformers import LlamaModel
from paddlenlp.transformers.llama.tokenizer_fast import LlamaTokenizerFast
from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from ppdiffusers.utils import export_to_video,export_to_video_2
from forwards import taylorstepfirstpredicthunyuanpipeline,taylorseer_step_firstpredict_hunyuan_forward
import time
#,taylorseer_hunyuan_single_block_forward,taylorseer_hunyuan_double_block_forward

os.environ["SKIP_PARENT_CLASS_CHECK"] = "True"
model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(model_id, subfolder="transformer", paddle_dtype=paddle.bfloat16)
tokenizer = LlamaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", dtype="float16")
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer = transformer,
    text_encoder = text_encoder,
    tokenizer = tokenizer,
    paddle_dtype=paddle.float16,
    map_location="cpu")
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.__class__.__call__ = taylorstepfirstpredicthunyuanpipeline
pipe.transformer.__class__.forward = taylorseer_step_firstpredict_hunyuan_forward
pipe.transformer.cnt = 0
pipe.transformer.num_steps = 50
pipe.transformer.predict_loss  = None
pipe.transformer.threshold= 0.12


# for double_transformer_block in pipe.transformer.transformer_blocks:
#     double_transformer_block.__class__.forward = taylorseer_hunyuan_double_block_forward
    
# for single_transformer_block in pipe.transformer.single_transformer_blocks:
#     single_transformer_block.__class__.forward = taylorseer_hunyuan_single_block_forward
for i in range(2):
    prompt = 'A cat walks on the grass, realistic.'
    start = time.time()
    output = pipe(
        prompt=prompt,
        height=480,
        width=640,
        num_frames=65,
        num_inference_steps=50,
        # num_videos_per_prompt=5
        generator=paddle.Generator().manual_seed(42),
    ).frames[0]
    elapsed1 = time.time() - start
    print(f"第一次运行时间: {elapsed1:.2f}s")

    export_to_video_2(output, "./text_to_video_generation-hunyuan_video.mp4", fps=24)