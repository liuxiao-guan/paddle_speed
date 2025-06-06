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

python infer_cogvideox_t2v_vctrl_cli.py \
  --pretrained_model_name_or_path "paddlemix/cogvideox-5b-vctrl" \
  --vctrl_path "paddlemix/vctrl-5b-t2v-canny" \
  --vctrl_config "vctrl_configs/cogvideox_5b_vctrl_config.json" \
  --control_video_path "examples/canny/case1/guide_values.mp4" \
  --output_dir "infer_outputs/canny2video/t2v" \
  --prompt_path "examples/canny/case1/prompt.txt" \
  --task "canny" \
  --width 720 \
  --height 480 \
  --max_frame 49 \
  --guidance_scale 3.5 \
  --num_inference_steps 25
