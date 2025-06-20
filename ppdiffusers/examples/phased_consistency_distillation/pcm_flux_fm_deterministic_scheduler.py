# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
import math
import paddle

from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.utils import BaseOutput, logging
from ppdiffusers.utils.paddle_utils import randn_tensor
from ppdiffusers.schedulers.scheduling_utils import SchedulerMixin


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FLUXPCMFMDeterministicSchedulerOutput(BaseOutput):
    prev_sample: paddle.Tensor


class FLUXPCMFMDeterministicScheduler(SchedulerMixin, ConfigMixin):

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        pcm_timesteps: int = 50,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
        invert_sigmas: bool = False,
        shift_terminal: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        time_shift_type: str = "exponential",
        stochastic_sampling: bool = False,
    ):
        timesteps = np.linspace(
            1, num_train_timesteps, num_train_timesteps, dtype=np.float32
        )[::-1].copy()
        timesteps = paddle.to_tensor(timesteps).to(dtype=paddle.float32)
        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.euler_timesteps = (
            np.arange(1, pcm_timesteps + 1) * (num_train_timesteps // pcm_timesteps)
        ).round().astype(np.int64) - 1
        self.sigmas = sigmas.numpy()[::-1][self.euler_timesteps]
        self.sigmas = paddle.to_tensor((self.sigmas[::-1].copy()))
        self.timesteps = self.sigmas * num_train_timesteps
        self._step_index = None
        self._begin_index = None
        self._shift = shift
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def shift(self):
        """
        The value used for shifting.
        """
        return self._shift

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_shift(self, shift: float):
        self._shift = shift

    def scale_noise(
        self,
        sample: paddle.Tensor,
        timestep: Union[float, paddle.Tensor],
        noise: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """
        Forward process in flow-matching

        Args:
            sample (`paddle.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `paddle.Tensor`:
                A scaled input sample.
        """
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.astype(dtype=sample.dtype)

        # TODO
        schedule_timesteps = self.timesteps

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: paddle.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def stretch_shift_to_terminal(self, t: paddle.Tensor) -> paddle.Tensor:
        r"""
        Stretches and shifts the timestep schedule to ensure it terminates at the configured `shift_terminal` config
        value.

        Reference:
        https://github.com/Lightricks/LTX-Video/blob/a01a171f8fe3d99dce2728d60a73fecf4d4238ae/ltx_video/schedulers/rf.py#L51

        Args:
            t (`torch.Tensor`):
                A tensor of timesteps to be stretched and shifted.

        Returns:
            `torch.Tensor`:
                A tensor of adjusted timesteps such that the final value equals `self.config.shift_terminal`.
        """
        one_minus_z = 1 - t
        scale_factor = one_minus_z[-1] / (1 - self.config.shift_terminal)
        stretched_t = 1 - (one_minus_z / scale_factor)
        return stretched_t

    def set_timesteps(
        self, 
        num_inference_steps: int = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")
        
        # if sigmas is None:
        #     timesteps = np.linspace(
        #         self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
        #     )

        #     sigmas = timesteps / self.config.num_train_timesteps
        # else:
        #     sigmas = np.array(sigmas).astype(np.float32)
        #     num_inference_steps = len(sigmas)

        num_inference_steps = 4
        timesteps = np.linspace(
            1, 1000, 1000, dtype=np.float32
        )[::-1].copy()
        timesteps = paddle.to_tensor(timesteps).to(dtype=paddle.float32)
        sigmas = timesteps / self.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        
        if self.config.shift_terminal:
            sigmas = self.stretch_shift_to_terminal(sigmas)

        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)

        # timesteps = np.linspace(
        #     self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
        # )
    
        sigmas = paddle.to_tensor(sigmas).astype(dtype=paddle.float32)
        # timesteps = sigmas * self.config.num_train_timesteps

        # set self.sigmas_ and timesteps
        self.sigmas = sigmas.numpy()[::-1][self.euler_timesteps]
        self.sigmas = paddle.to_tensor((self.sigmas[::-1].copy()))
        inference_indices = np.linspace(
            0, self.config.pcm_timesteps, num=num_inference_steps, endpoint=False
        )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = paddle.to_tensor(inference_indices).astype("int64")
        self.sigmas_ = self.sigmas[inference_indices]
        timesteps = self.sigmas_ * self.config.num_train_timesteps
        # self.sigmas_ = paddle.concat(x=[self.sigmas_, paddle.zeros(shape=[1])])

        if self.config.invert_sigmas:
            self.sigmas_ = 1.0 - self.sigmas_
            timesteps = self.sigmas_ * self.config.num_train_timesteps
            self.sigmas_ = paddle.concat([self.sigmas_, paddle.ones(1)])
        else:
            self.sigmas_ = paddle.concat([self.sigmas_, paddle.zeros(1)])
        print(self.sigmas_)

        self.timesteps = timesteps
        # self.sigmas = sigmas
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, paddle.Tensor):
                timestep = timestep
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: paddle.Tensor,
        timestep: Union[float, paddle.Tensor],
        sample: paddle.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[paddle.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FLUXPCMFMDeterministicSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`paddle.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`paddle.Tensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`paddle.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if isinstance(timestep, int) or (isinstance(timestep, paddle.Tensor) and "int" in str(timestep.dtype)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(paddle.float32)

        sigma = self.sigmas_[self.step_index]

        denoised = sample - model_output * sigma
        derivative = (sample - denoised) / sigma

        dt = self.sigmas_[self.step_index + 1] - sigma
        prev_sample = sample + derivative * dt
        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FLUXPCMFMDeterministicSchedulerOutput(prev_sample=prev_sample)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_exponential
    def _convert_to_exponential(self, in_sigmas: paddle.Tensor, num_inference_steps: int) -> paddle.Tensor:
        """Constructs an exponential noise schedule."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_beta
    def _convert_to_beta(
        self, in_sigmas: paddle.Tensor, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6
    ) -> paddle.Tensor:
        """From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)"""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return sigmas

    def __len__(self):
        return self.config.num_train_timesteps