# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 Katherine Crowson and The HuggingFace Team. All rights reserved.
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
import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
from scipy import integrate

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ..scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
# Copied from ppdiffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->LMSDiscrete
class PreconfigLMSDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: paddle.Tensor
    pred_original_sample: Optional[paddle.Tensor] = None


# Copied from ppdiffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return paddle.to_tensor(betas, dtype=paddle.float32)


class PreconfigLMSDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Linear Multistep Scheduler for discrete beta schedules. Based on the original k-diffusion implementation by
    Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        preconfig=True,
    ):
        if trained_betas is not None:
            self.betas = paddle.to_tensor(trained_betas, dtype=paddle.float32)
        elif beta_schedule == "linear":
            self.betas = paddle.linspace(beta_start, beta_end, num_train_timesteps, dtype=paddle.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                paddle.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=paddle.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = paddle.cumprod(self.alphas, 0)

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        self.sigmas = paddle.to_tensor(sigmas)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = self.sigmas.max()

        # setable values
        self.num_inference_steps = None
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        self.timesteps = paddle.to_tensor(timesteps, dtype=paddle.float32)
        self.derivatives = []
        self.is_scale_input_called = False
        self.preconfig = preconfig

    def scale_model_input(
        self, sample: paddle.Tensor, timestep: Union[float, paddle.Tensor], **kwargs
    ) -> paddle.Tensor:
        """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the K-LMS algorithm.

        Args:
            sample (`paddle.Tensor`): input sample
            timestep (`float` or `paddle.Tensor`): the current timestep in the diffusion chain

        Returns:
            `paddle.Tensor`: scaled input sample
        """
        if kwargs.get("step_index") is not None:
            step_index = kwargs["step_index"]
        else:
            step_index = (self.timesteps == timestep).nonzero().item()
        self.is_scale_input_called = True
        if not self.preconfig:
            sigma = self.sigmas[step_index]
            sample = sample / ((sigma**2 + 1) ** 0.5)
            return sample
        else:
            return sample * self.latent_scales[step_index]

    def get_lms_coefficient(self, order, t, current_order):
        """
        Compute a linear multistep coefficient.

        Args:
            order (TODO):
            t (TODO):
            current_order (TODO):
        """

        def lms_derivative(tau):
            prod = 1.0
            for k in range(order):
                if current_order == k:
                    continue
                prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
            return prod

        integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]

        return integrated_coeff

    def set_timesteps(self, num_inference_steps: int, preconfig_order: int = 4):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = paddle.to_tensor(sigmas)
        self.timesteps = paddle.to_tensor(timesteps, dtype=paddle.float32)

        self.derivatives = []
        if self.preconfig:
            self.order = preconfig_order
            self.lms_coeffs = []
            self.latent_scales = [1.0 / ((sigma**2 + 1) ** 0.5) for sigma in self.sigmas]
            for step_index in range(self.num_inference_steps):
                order = min(step_index + 1, preconfig_order)
                self.lms_coeffs.append(
                    [self.get_lms_coefficient(order, step_index, curr_order) for curr_order in range(order)]
                )

    def step(
        self,
        model_output: paddle.Tensor,
        timestep: Union[float, paddle.Tensor],
        sample: paddle.Tensor,
        order: int = 4,
        return_dict: bool = True,
        **kwargs
    ) -> Union[PreconfigLMSDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`paddle.Tensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                current instance of sample being created by diffusion process.
            order: coefficient for multi-step inference.
            return_dict (`bool`): option for returning tuple rather than PreconfigLMSDiscreteSchedulerOutput class
            Args in kwargs:
                step_index (`int`):
                return_pred_original_sample (`bool`): option for return pred_original_sample

        Returns:
            [`~schedulers.scheduling_utils.PreconfigLMSDiscreteSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.PreconfigLMSDiscreteSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.

        """
        if not self.is_scale_input_called:
            warnings.warn(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )
        if kwargs.get("return_pred_original_sample") is not None:
            return_pred_original_sample = kwargs["return_pred_original_sample"]
        else:
            return_pred_original_sample = True
        if kwargs.get("step_index") is not None:
            step_index = kwargs["step_index"]
        else:
            step_index = (self.timesteps == timestep).nonzero().item()
        if self.config.prediction_type == "epsilon" and not return_pred_original_sample:
            # if pred_original_sample is no need
            self.derivatives.append(model_output)
            pred_original_sample = None
        else:
            sigma = self.sigmas[step_index]
            # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            if self.config.prediction_type == "epsilon":
                pred_original_sample = sample - sigma * model_output
            elif self.config.prediction_type == "v_prediction":
                # * c_out + input * c_skip
                pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            elif self.config.prediction_type == "sample":
                pred_original_sample = model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
                )
            # 2. Convert to an ODE derivative
            derivative = (sample - pred_original_sample) / sigma
            self.derivatives.append(derivative)

        if len(self.derivatives) > order:
            self.derivatives.pop(0)

        if not self.preconfig:
            # 3. If not preconfigured, compute linear multistep coefficients.
            order = min(step_index + 1, order)
            lms_coeffs = [self.get_lms_coefficient(order, step_index, curr_order) for curr_order in range(order)]
            # 4. Compute previous sample based on the derivatives path
            prev_sample = sample + sum(
                coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(self.derivatives))
            )
        else:
            # 3. If preconfigured, direct compute previous sample based on the derivatives path
            prev_sample = sample + sum(
                coeff * derivative
                for coeff, derivative in zip(self.lms_coeffs[step_index], reversed(self.derivatives))
            )

        if not return_dict:
            if not return_pred_original_sample:
                return (prev_sample,)
            else:
                return (prev_sample, pred_original_sample)

        return PreconfigLMSDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
    ) -> paddle.Tensor:
        # Fix 0D tensor
        if paddle.is_tensor(timesteps) and timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        # Make sure sigmas and timesteps have the same dtype as original_samples
        sigmas = self.sigmas.cast(original_samples.dtype)
        schedule_timesteps = self.timesteps

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
