


from typing import Any, Callable, Dict, List, Optional, Tuple, Union


import paddle

from ppdiffusers.pipelines.pipeline_utils import ImagePipelineOutput
from ppdiffusers.utils.paddle_utils import randn_tensor
from cache_functions import cache_init,cal_type
try:
    # paddle.incubate.jit.inference is available in paddle develop but not in paddle 3.0beta, so we add a try except.
    from paddle.incubate.jit import is_inference_mode
except:

    def is_inference_mode(func):
        return False
@paddle.no_grad()
def taylorseer_dit_pipeline(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`paddle.Generator`, *optional*):
                A [`paddle.Generator`] to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        ```py
        >>> from ppdiffusers import DiTPipeline, DPMSolverMultistepScheduler
        >>> import paddle

        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", paddle_dtype=paddle.float16)
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        >>> # pick words from Imagenet class labels
        >>> pipe.labels  # to print all available words

        >>> # pick words that exist in ImageNet
        >>> words = ["white shark", "umbrella"]

        >>> class_ids = pipe.get_label_ids(words)

        >>> generator = paddle.Generator().manual_seed(33)
        >>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

        >>> image = output.images[0]  # label 'white shark'
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            dtype=self.transformer.dtype,
        )
        latent_model_input = paddle.concat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = paddle.to_tensor(class_labels).reshape(
            [
                -1,
            ]
        )
        class_null = paddle.to_tensor([1000] * batch_size)
        class_labels_input = paddle.concat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels
        cache_dic, current= cache_init(num_inference_steps)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for idx, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = paddle.concat([half, half], axis=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            current['step']  = idx
            timesteps = t

            if not paddle.is_tensor(timesteps):
                if isinstance(timesteps, float):
                    dtype = paddle.float32
                else:
                    dtype = paddle.int64
                timesteps = paddle.to_tensor([timesteps], dtype=dtype)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None]
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(
                [
                    latent_model_input.shape[0],
                ]
            )
           
            # predict noise model_output
            noise_pred_out = self.transformer(latent_model_input, timestep=timesteps, class_labels=class_labels_input,cache_dic=cache_dic,current=current)
            if is_inference_mode(self.transformer):
                # self.transformer run in paddle inference.
                noise_pred = noise_pred_out
            else:
                noise_pred = noise_pred_out.sample

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = paddle.chunk(eps, 2, axis=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = paddle.concat([half_eps, half_eps], axis=0)

                noise_pred = paddle.concat([eps, rest], axis=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = paddle.split(
                    noise_pred, [latent_channels, noise_pred.shape[1] - latent_channels], axis=1
                )
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, axis=0)
        else:
            latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents

        samples_out = self.vae.decode(latents)
        if is_inference_mode(self.vae.decode):
            # self.vae.decode run in paddle inference.
            samples = samples_out
        else:
            samples = samples_out.sample

        samples = (samples / 2 + 0.5).clip(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.transpose([0, 2, 3, 1]).cast("float32").cpu().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)