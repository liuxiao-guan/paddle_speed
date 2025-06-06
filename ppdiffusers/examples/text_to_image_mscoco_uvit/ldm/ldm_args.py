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

import types
from dataclasses import asdict, dataclass, field
from typing import Optional

import paddle
from paddlenlp.trainer import TrainingArguments
from paddlenlp.utils.log import logger


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    # use pretrained vae kl-8.ckpt (CompVis/stable-diffusion-v1-4/vae)
    vae_name_or_path: Optional[str] = field(
        default="CompVis/stable-diffusion-v1-4/vae",
        metadata={"help": "pretrained_vae_name_or_path"},
    )
    text_encoder_name_or_path: Optional[str] = field(
        default="runwayml/stable-diffusion-v1-5/text_encoder",
        metadata={"help": "pretrained_text_encoder_name_or_path"},
    )
    uvit_config_file: Optional[str] = field(
        default="./config/config/uvit_t2i_small.json", metadata={"help": "uvit_config_file"}
    )
    tokenizer_name: Optional[str] = field(
        default="openai/clip-vit-large-patch14",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    model_max_length: Optional[int] = field(default=77, metadata={"help": "Pretrained tokenizer model_max_length"})
    num_inference_steps: Optional[int] = field(default=50, metadata={"help": "num_inference_steps"})
    train_text_encoder: bool = field(default=False, metadata={"help": "Whether or not train text encoder"})

    use_ema: bool = field(default=False, metadata={"help": "Whether or not use ema"})
    pretrained_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model, when we want to resume training."},
    )
    image_logging_steps: Optional[int] = field(default=1000, metadata={"help": "Log image every X steps."})
    enable_xformers_memory_efficient_attention: bool = field(
        default=False, metadata={"help": "enable_xformers_memory_efficient_attention."}
    )
    only_save_updated_model: bool = field(
        default=True, metadata={"help": "Whether or not save only_save_updated_model"}
    )
    unet_learning_rate: float = field(default=None, metadata={"help": "The initial learning rate for Unet Model."})
    text_encoder_learning_rate: float = field(
        default=None,
        metadata={"help": "The initial learning rate for Text Encoder Model."},
    )

    # to_static: bool = field(default=False, metadata={"help": "Whether or not to_static"})
    prediction_type: Optional[str] = field(
        default="epsilon",
        metadata={
            "help": "prediction_type, prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)"
        },
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark."},
    )
    profiler_options: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    noise_offset: Optional[int] = field(default=0, metadata={"help": "The scale of noise offset."})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """

    feature_path: str = field(
        default="./datasets/coco256_features/",
        metadata={"help": "The feature path."},
    )
    resolution: int = field(
        default=256,
        metadata={
            "help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        },
    )
    num_records: int = field(default=10000000, metadata={"help": "num_records"})
    buffer_size: int = field(
        default=100,
        metadata={"help": "Buffer size"},
    )
    shuffle_every_n_samples: int = field(
        default=5,
        metadata={"help": "shuffle_every_n_samples."},
    )


@dataclass
class TrainerArguments(TrainingArguments):
    """
    Arguments pertaining to what training options we are going to use during pretraining.
    """

    pretrained_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Whether to use pretrained checkpoint weights."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )

    optim: str = field(default="adamw", metadata={"help": "optimizer setting, [lamb/adamw]"})
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=0.0, metadata={"help": "Max gradient norm."})  # clip_grad

    # new added
    warmup_lr: float = field(default=0.0, metadata={"help": "The initial learning rate for AdamW."})
    min_lr: float = field(default=0.0, metadata={"help": "The initial learning rate for AdamW."})
    warmup_steps: int = field(default=-1, metadata={"help": "Linear warmup over warmup_steps."})
    warmup_epochs: int = field(default=1, metadata={"help": "Linear warmup over warmup_epochs."})

    output_dir: str = field(
        default="output_dir",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    logging_dir: str = field(
        default="output_dir/tb_ft_log",
        metadata={"help": "The output directory where logs saved."},
    )
    logging_steps: int = field(default=10, metadata={"help": "logging_steps print frequency (default: 10)"})

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    do_export: bool = field(default=False, metadata={"help": "Whether to export infernece model."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU core/CPU for training."})
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    accum_freq: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    num_train_epochs: float = field(default=-1, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use. support linear, cosine, constant, constant_with_warmup"},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    num_cycles: float = field(default=0.5, metadata={"help": "The number of waves in the cosine scheduler."})
    lr_end: float = field(default=1e-7, metadata={"help": "The end LR in the polynomial scheduler."})
    power: float = field(default=1.0, metadata={"help": "The power factor in the polynomial scheduler."})

    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_epochs: int = field(default=1, metadata={"help": "Save checkpoint every X updates epochs."})

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (no_cuda). This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to Use fp16 (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: AMP optimization level selected in ['O0', 'O1', and 'O2']. "
                "See details at https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html"
            )
        },
    )

    dp_degree: int = field(
        default=1,
        metadata={"help": " data parallel degrees."},
    )
    sharding_parallel_degree: int = field(
        default=1,
        metadata={"help": " sharding parallel degrees."},
    )
    tensor_parallel_degree: int = field(
        default=1,
        metadata={"help": " tensor parallel degrees."},
    )
    pipeline_parallel_degree: int = field(
        default=1,
        metadata={"help": " pipeline parallel degrees."},
    )
    sep_parallel_degree: int = field(
        default=1,
        metadata={"help": ("sequence parallel strategy.")},
    )

    last_epoch: int = field(default=-1, metadata={"help": "the last epoch to resume"})

    dataloader_drop_last: bool = field(
        default=True, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    dataloader_num_workers: int = field(
        default=1,
        metadata={
            "help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )

    disable_tqdm: Optional[bool] = field(
        default=True, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )
    tensorboard: bool = field(
        default=False,
        metadata={"help": "Whether to use tensorboard to record loss."},
    )


@dataclass
class NoTrainerTrainingArguments:
    output_dir: str = field(
        default="outputs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU core/CPU for training."}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=2e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.03, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.9, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=-1.0, metadata={"help": "Max gradient norm."})
    num_train_epochs: int = field(default=1000000, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=1000000,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: str = field(
        default="constant_with_warmup",
        metadata={
            "help": 'The scheduler type to use. support ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
        },
    )
    warmup_steps: int = field(default=5000, metadata={"help": "Linear warmup over warmup_steps."})

    logging_dir: Optional[str] = field(default="logs", metadata={"help": "VisualDL log dir."})

    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})

    save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})

    seed: int = field(
        default=1234,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    dataloader_num_workers: int = field(
        default=16,
        metadata={
            "help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "The list of integrations to report the results and logs to."},
    )
    recompute: bool = field(
        default=False,
        metadata={
            "help": "Recompute the forward pass to calculate gradients. Used for saving memory. "
            "Only support for networks with transformer blocks."
        },
    )
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (no_cuda). This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to Use fp16 (mixed) precision instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: AMP optimization level selected in ['O0', 'O1', and 'O2']. "
                "See details at https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html"
            )
        },
    )

    def __str__(self):
        self_as_dict = asdict(self)
        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    def print_config(self, args=None, key=""):
        """
        print all config values.
        """
        logger.info("=" * 60)
        if args is None:
            args = self
            key = "Training"

        logger.info("{:^40}".format("{} Configuration Arguments".format(key)))
        logger.info("{:30}:{}".format("paddle commit id", paddle.version.commit))

        for a in dir(args):
            if a[:2] != "__":  # don't print double underscore methods
                v = getattr(args, a)
                if not isinstance(v, types.MethodType):
                    logger.info("{:30}:{}".format(a, v))

        logger.info("")
