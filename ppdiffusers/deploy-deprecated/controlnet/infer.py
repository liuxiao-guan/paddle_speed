# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os
import time

# isort: split
import paddle

# isort: split
import cv2
import fastdeploy as fd
import numpy as np
from paddlenlp.trainer.argparser import strtobool
from PIL import Image
from tqdm.auto import trange

from ppdiffusers import DiffusionPipeline, FastDeployStableDiffusionMegaPipeline
from ppdiffusers.utils import load_image


def get_canny_image(image, args):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = cv2.Canny(image, args.low_threshold, args.high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="runwayml/stable-diffusion-v1-5@fastdeploy",
        help="The model directory of diffusion_model.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
        help="The number of unet inference steps.",
    )
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=1,
        help="The number of performance benchmark steps.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle_tensorrt",
        # Note(zhoushunjie): Will support 'tensorrt' soon.
        choices=["onnx_runtime", "paddle", "paddlelite", "paddle_tensorrt"],
        help="The inference runtime backend of unet model and text encoder model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        # Note(shentanyue): Will support more devices.
        choices=[
            "cpu",
            "gpu",
            "huawei_ascend_npu",
            "kunlunxin_xpu",
        ],
        help="The inference runtime device of models.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="text2img_control",
        choices=[
            "text2img_control",
            "img2img_control",
            "inpaint_legacy_control",
            "hiresfix_control",
            "all",
        ],
        help="The task can be one of [text2img_control, img2img_control, inpaint_legacy_control, hiresfix_control, all]. ",
    )
    parser.add_argument(
        "--parse_prompt_type",
        type=str,
        default="lpw",
        choices=[
            "raw",
            "lpw",
        ],
        help="The parse_prompt_type can be one of [raw, lpw]. ",
    )
    parser.add_argument("--use_fp16", type=strtobool, default=True, help="Whether to use FP16 mode")
    parser.add_argument("--device_id", type=int, default=0, help="The selected gpu id. -1 means use cpu")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="preconfig-euler-ancestral",
        choices=[
            "pndm",
            "lms",
            "euler",
            "euler-ancestral",
            "preconfig-euler-ancestral",
            "dpm-multi",
            "dpm-single",
            "unipc-multi",
            "ddim",
            "ddpm",
            "deis-multi",
            "heun",
            "kdpm2-ancestral",
            "kdpm2",
        ],
        help="The scheduler type of stable diffusion.",
    )
    parser.add_argument(
        "--infer_op",
        type=str,
        default="zero_copy_infer",
        choices=[
            "zero_copy_infer",
            "raw",
            "all",
        ],
        help="The type of infer op.",
    )
    parser.add_argument("--height", type=int, default=512, help="Height of input image")
    parser.add_argument("--width", type=int, default=512, help="Width of input image")
    parser.add_argument("--hr_resize_height", type=int, default=768, help="HR Height of input image")
    parser.add_argument("--hr_resize_width", type=int, default=768, help="HR Width of input image")
    parser.add_argument("--is_sd2_0", type=strtobool, default=False, help="Is sd2_0 model?")
    parser.add_argument(
        "--low_threshold",
        type=int,
        default=100,
        help="The value of Canny low threshold.",
    )
    parser.add_argument(
        "--high_threshold",
        type=int,
        default=200,
        help="The value of Canny high threshold.",
    )
    return parser.parse_args()


def create_ort_runtime(device_id=0):
    option = fd.RuntimeOption()
    option.use_ort_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
    return option


def create_paddle_inference_runtime(
    use_trt=False,
    dynamic_shape=None,
    use_fp16=False,
    device_id=0,
    disable_paddle_trt_ops=[],
    disable_paddle_pass=[],
    paddle_stream=None,
    workspace=None,
):
    option = fd.RuntimeOption()
    option.use_paddle_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
    if paddle_stream is not None and use_trt:
        option.set_external_raw_stream(paddle_stream)
    for pass_name in disable_paddle_pass:
        option.paddle_infer_option.delete_pass(pass_name)
    if use_trt:
        option.paddle_infer_option.disable_trt_ops(disable_paddle_trt_ops)
        option.paddle_infer_option.enable_trt = True
        if workspace is not None:
            option.set_trt_max_workspace_size(workspace)
        if use_fp16:
            option.trt_option.enable_fp16 = True
        else:
            # Note(zhoushunjie): These four passes don't support fp32 now.
            # Remove this line of code in future.
            only_fp16_passes = [
                "trt_cross_multihead_matmul_fuse_pass",
                "trt_flash_multihead_matmul_fuse_pass",
                "preln_elementwise_groupnorm_act_pass",
                "elementwise_groupnorm_act_pass",
            ]
            for curr_pass in only_fp16_passes:
                option.paddle_infer_option.delete_pass(curr_pass)

        # Need to enable collect shape
        if dynamic_shape is not None:
            option.paddle_infer_option.collect_trt_shape = True
            for key, shape_dict in dynamic_shape.items():
                option.trt_option.set_shape(
                    key,
                    shape_dict["min_shape"],
                    shape_dict.get("opt_shape", None),
                    shape_dict.get("max_shape", None),
                )
    return option


def create_paddle_lite_runtime(device="cpu", device_id=0, use_fp16=False):
    option = fd.RuntimeOption()
    option.use_paddle_lite_backend()
    if device == "huawei_ascend_npu":
        option.use_ascend()
        option.set_lite_device_names(["huawei_ascend_npu"])
        option.set_lite_context_properties(
            "HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS={};HUAWEI_ASCEND_NPU_PRECISION_MODE=allow_mix_precision".format(
                device_id
            )
        )
    elif device == "kunlunxin_xpu":
        # TODO(shentanyue): Add kunlunxin_xpu code
        # https://github.com/PaddlePaddle/FastDeploy/blob/4c3e7030e151528d304619901c794481bb2f6037/examples/multimodal/stable_diffusion/infer.py#L178-L195
        option.use_kunlunxin(
            device_id,
            l3_workspace_size=(64 * 1024 * 1024 - 4 * 1024),
            locked=False,
            autotune=False,
            autotune_file="",
            precision="int16",
            adaptive_seqlen=True,
            enable_multi_stream=True,
        )
        if use_fp16:
            option.enable_lite_fp16()
    else:
        pass
    return option


def create_trt_runtime(workspace=(1 << 31), dynamic_shape=None, use_fp16=False, device_id=0):
    option = fd.RuntimeOption()
    option.use_trt_backend()
    option.use_gpu(device_id)
    if use_fp16:
        option.enable_trt_fp16()
    if workspace is not None:
        option.set_trt_max_workspace_size(workspace)
    if dynamic_shape is not None:
        for key, shape_dict in dynamic_shape.items():
            option.set_trt_input_shape(
                key,
                min_shape=shape_dict["min_shape"],
                opt_shape=shape_dict.get("opt_shape", None),
                max_shape=shape_dict.get("max_shape", None),
            )
    # cache_file = os.path.join(model_dir, model_prefix, "inference.trt")
    # option.set_trt_cache_file(cache_file)
    return option


def main(args):
    if args.device_id == -1:
        paddle.set_device("cpu")
        paddle_stream = None
    else:
        paddle.set_device(f"gpu:{args.device_id}")
        paddle_stream = paddle.device.cuda.current_stream(args.device_id).cuda_stream

    infer_op_dict = {
        "vae_encoder": args.infer_op,
        "vae_decoder": args.infer_op,
        "text_encoder": args.infer_op,
        "unet": args.infer_op,
    }
    seed = 1024
    vae_in_channels = 4
    text_encoder_max_length = 77
    unet_max_length = text_encoder_max_length * 3  # lpw support max_length is 77x3
    min_image_size = 512
    max_image_size = 768
    max_image_size = max(min_image_size, max_image_size)
    hidden_states = 1024 if args.is_sd2_0 else 768
    unet_in_channels = 4

    text_encoder_dynamic_shape = {
        "input_ids": {
            "min_shape": [1, text_encoder_max_length],
            "max_shape": [1, text_encoder_max_length],
            "opt_shape": [1, text_encoder_max_length],
        }
    }
    vae_encoder_dynamic_shape = {
        "sample": {
            "min_shape": [1, 3, min_image_size, min_image_size],
            "max_shape": [1, 3, max_image_size, max_image_size],
            "opt_shape": [1, 3, min_image_size, min_image_size],
        }
    }
    vae_decoder_dynamic_shape = {
        "latent_sample": {
            "min_shape": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
            "max_shape": [1, vae_in_channels, max_image_size // 8, max_image_size // 8],
            "opt_shape": [1, vae_in_channels, min_image_size // 8, min_image_size // 8],
        }
    }
    unet_dynamic_shape = {
        "sample": {
            "min_shape": [
                1,
                unet_in_channels,
                min_image_size // 8,
                min_image_size // 8,
            ],
            "max_shape": [
                2,
                unet_in_channels,
                max_image_size // 8,
                max_image_size // 8,
            ],
            "opt_shape": [
                2,
                unet_in_channels,
                min_image_size // 8,
                min_image_size // 8,
            ],
        },
        "timestep": {
            "min_shape": [1],
            "max_shape": [1],
            "opt_shape": [1],
        },
        "encoder_hidden_states": {
            "min_shape": [1, text_encoder_max_length, hidden_states],
            "max_shape": [2, unet_max_length, hidden_states],
            "opt_shape": [2, text_encoder_max_length, hidden_states],
        },
        "controlnet_cond": {
            "min_shape": [1, 3, min_image_size, min_image_size],
            "max_shape": [2, 3, max_image_size, max_image_size],
            "opt_shape": [2, 3, min_image_size, min_image_size],
        },
        "controlnet_conditioning_scale": {
            "min_shape": [13],
            "max_shape": [13],
            "opt_shape": [13],
        },
    }
    # 4. Init runtime
    if args.backend == "onnx_runtime":
        runtime_options = dict(
            text_encoder=create_ort_runtime(device_id=args.device_id),
            vae_encoder=create_ort_runtime(device_id=args.device_id),
            vae_decoder=create_ort_runtime(device_id=args.device_id),
            unet=create_ort_runtime(device_id=args.device_id),
        )
    elif args.backend == "paddlelite":
        runtime_options = dict(
            text_encoder=create_paddle_lite_runtime(device=args.device, device_id=args.device_id, use_fp16=False),
            vae_encoder=create_paddle_lite_runtime(device=args.device, device_id=args.device_id, use_fp16=False),
            vae_decoder=create_paddle_lite_runtime(device=args.device, device_id=args.device_id, use_fp16=False),
            unet=create_paddle_lite_runtime(device=args.device, device_id=args.device_id, use_fp16=args.use_fp16),
        )
    elif args.backend == "tensorrt":
        runtime_options = dict(
            text_encoder=create_trt_runtime(
                dynamic_shape=text_encoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
            ),
            vae_encoder=create_trt_runtime(
                dynamic_shape=vae_encoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
            ),
            vae_decoder=create_trt_runtime(
                dynamic_shape=vae_decoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
            ),
            unet=create_trt_runtime(
                dynamic_shape=unet_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
            ),
        )
    elif args.backend == "paddle" or args.backend == "paddle_tensorrt":
        args.use_trt = args.backend == "paddle_tensorrt"
        runtime_options = dict(
            text_encoder=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=text_encoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
                disable_paddle_trt_ops=["arg_max", "range", "lookup_table_v2"],
                paddle_stream=paddle_stream,
            ),
            vae_encoder=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=vae_encoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
                paddle_stream=paddle_stream,
            ),
            vae_decoder=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=vae_decoder_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
                paddle_stream=paddle_stream,
            ),
            unet=create_paddle_inference_runtime(
                use_trt=args.use_trt,
                dynamic_shape=unet_dynamic_shape,
                use_fp16=args.use_fp16,
                device_id=args.device_id,
                paddle_stream=paddle_stream,
            ),
        )
    pipe = FastDeployStableDiffusionMegaPipeline.from_pretrained(
        args.model_dir,
        runtime_options=runtime_options,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.change_scheduler(args.scheduler)
    parse_prompt_type = args.parse_prompt_type
    width = args.width
    height = args.height
    hr_resize_width = args.hr_resize_width
    hr_resize_height = args.hr_resize_height

    if args.infer_op == "all":
        infer_op_list = ["zero_copy_infer", "raw"]
    else:
        infer_op_list = [args.infer_op]
    if args.device == "kunlunxin_xpu" or args.backend == "paddle":
        print("When device is kunlunxin_xpu or backend is paddle, we will use `raw` infer op.")
        infer_op_list = ["raw"]

    for infer_op in infer_op_list:
        infer_op_dict = {
            "vae_encoder": infer_op,
            "vae_decoder": infer_op,
            "text_encoder": infer_op,
            "unet": infer_op,
        }
        folder = f"infer_op_{infer_op}_fp16" if args.use_fp16 else f"infer_op_{infer_op}_fp32"
        os.makedirs(folder, exist_ok=True)

        if args.task_name in ["text2img_control", "all"]:
            init_image = load_image(
                "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny_demo.png"
            )
            controlnet_cond = get_canny_image(init_image, args)
            # text2img
            prompt = "bird"
            time_costs = []
            # warmup
            pipe.text2img(
                prompt,
                num_inference_steps=10,
                height=height,
                width=width,
                parse_prompt_type=parse_prompt_type,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                infer_op_dict=infer_op_dict,
            )
            print("==> Test text2img_control performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.text2img(
                    prompt,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    parse_prompt_type=parse_prompt_type,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
                    infer_op_dict=infer_op_dict,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/text2img_control.png")

        if args.task_name in ["img2img_control", "all"]:
            img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/sketch-mountains-input.png"
            init_image = load_image(img_url)
            controlnet_cond = get_canny_image(init_image, args)
            prompt = "A fantasy landscape, trending on artstation"
            time_costs = []
            # warmup
            pipe.img2img(
                prompt,
                image=init_image,
                num_inference_steps=20,
                height=height,
                width=width,
                parse_prompt_type=parse_prompt_type,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                infer_op_dict=infer_op_dict,
            )
            print("==> Test img2img_control performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.img2img(
                    prompt,
                    image=init_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    parse_prompt_type=parse_prompt_type,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
                    infer_op_dict=infer_op_dict,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/img2img_control.png")

        if args.task_name in ["inpaint_legacy_control", "all"]:
            img_url = (
                "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
            )
            mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"
            init_image = load_image(img_url)
            mask_image = load_image(mask_url)
            controlnet_cond = get_canny_image(init_image, args)
            prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
            time_costs = []

            pipe.inpaint_legacy(
                prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=20,
                height=height,
                width=width,
                parse_prompt_type=parse_prompt_type,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                infer_op_dict=infer_op_dict,
            )
            print("==> Test inpaint_legacy_control performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = pipe.inpaint_legacy(
                    prompt,
                    image=init_image,
                    mask_image=mask_image,
                    num_inference_steps=args.inference_steps,
                    height=height,
                    width=width,
                    parse_prompt_type=parse_prompt_type,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
                    infer_op_dict=infer_op_dict,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            if args.task_name == "all":
                task_name = "inpaint_legacy_control"
            else:
                task_name = args.task_name
            images[0].save(f"{folder}/{task_name}.png")

        if args.task_name in ["hiresfix_control", "all"]:
            hiresfix_pipe = DiffusionPipeline.from_pretrained(
                args.model_dir,
                vae_encoder=pipe.vae_encoder,
                vae_decoder=pipe.vae_decoder,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
                requires_safety_checker=pipe.requires_safety_checker,
                custom_pipeline="pipeline_fastdeploy_stable_diffusion_hires_fix",
            )
            # custom_pipeline
            # https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/examples/community/pipeline_fastdeploy_stable_diffusion_hires_fix.py
            hiresfix_pipe._progress_bar_config = pipe._progress_bar_config
            # hiresfix_control
            init_image = load_image(
                "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny_demo.png"
            )
            controlnet_cond = get_canny_image(init_image, args)
            # hiresfix_control
            prompt = "a red bird"
            time_costs = []
            # warmup
            hiresfix_pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=20,
                hires_ratio=0.5,
                hr_resize_width=hr_resize_width,
                hr_resize_height=hr_resize_height,
                enable_hr=True,
                controlnet_cond=controlnet_cond,
                controlnet_conditioning_scale=1.0,
                parse_prompt_type=parse_prompt_type,
                infer_op_dict=infer_op_dict,
            )
            print("==> Test hiresfix_control performance.")
            for step in trange(args.benchmark_steps):
                start = time.time()
                paddle.seed(seed)
                images = hiresfix_pipe(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=args.inference_steps,
                    hires_ratio=0.5,
                    hr_resize_width=hr_resize_width,
                    hr_resize_height=hr_resize_height,
                    enable_hr=True,
                    controlnet_cond=controlnet_cond,
                    controlnet_conditioning_scale=1.0,
                    parse_prompt_type=parse_prompt_type,
                    infer_op_dict=infer_op_dict,
                ).images
                latency = time.time() - start
                time_costs += [latency]
                # print(f"No {step:3d} time cost: {latency:2f} s")
            print(
                f"Mean latency: {np.mean(time_costs):2f} s, p50 latency: {np.percentile(time_costs, 50):2f} s, "
                f"p90 latency: {np.percentile(time_costs, 90):2f} s, p95 latency: {np.percentile(time_costs, 95):2f} s."
            )
            images[0].save(f"{folder}/hiresfix_control.png")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
