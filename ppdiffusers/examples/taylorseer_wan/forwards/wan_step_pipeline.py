

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle

from ppdiffusers.pipelines.wan.pipeline_output import WanPipelineOutput
# from .pipeline_output import HunyuanVideoPipelineOutput
# from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ppdiffusers.callbacks import MultiPipelineCallbacks,PipelineCallback
from cache_functions import cache_step_init
from ppdiffusers import WanPipeline

@paddle.no_grad()
# @replace_example_docstring(EXAMPLE_DOC_STRING)
def wan_step_pipeline(
    self:WanPipeline,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
    latents: Optional[paddle.Tensor] = None,
    prompt_embeds: Optional[paddle.Tensor] = None,
    negative_prompt_embeds: Optional[paddle.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        height (`int`, defaults to `480`):
            The height in pixels of the generated image.
        width (`int`, defaults to `832`):
            The width in pixels of the generated image.
        num_frames (`int`, defaults to `81`):
            The number of frames in the generated video.
        num_inference_steps (`int`, defaults to `50`):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, defaults to `5.0`):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
            A [`paddle.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`paddle.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`paddle.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
            A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
            each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
            DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
            list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        autocast_dtype (`paddle.dtype`, *optional*, defaults to `paddle.bfloat16`):
            The dtype to use for the paddle.amp.autocast.

    Examples:

    Returns:
        [`~WanPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
            the first element is a list with the generated images and the second element is a list of `bool`s
            indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
    """

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    device = None

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    transformer_dtype = self.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_frames,
        paddle.float32,
        device,
        generator,
        latents,
    )

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    cache_dic, current = cache_step_init(self._num_timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            
            current['step'] = i

            self._current_timestep = t
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0])

            # current_stream = "cond_stream"
            current['stream'] = "cond_stream"
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
                current = current,
                cache_dic = cache_dic 
            )[0]
            # current_stream = "uncond_stream"
            current['stream'] = "uncond_stream"
            if self.do_classifier_free_guidance:
                noise_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    current = current,
                    cache_dic = cache_dic 
                )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            # if XLA_AVAILABLE:
            #     xm.mark_step()

    self._current_timestep = None

    if not output_type == "latent":
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            paddle.to_tensor(self.vae.config.latents_mean)
            .view([1, self.vae.config.z_dim, 1, 1, 1])
            .to(latents.dtype)
        )
        latents_std = 1.0 / paddle.to_tensor(self.vae.config.latents_std).view(
            [1, self.vae.config.z_dim, 1, 1, 1]
        ).to(latents.dtype)
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return WanPipelineOutput(frames=video)