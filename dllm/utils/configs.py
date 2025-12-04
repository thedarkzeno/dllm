import os
from dataclasses import dataclass, field

import transformers

from dllm.utils.utils import resolve_with_base_env


@dataclass
class ModelArguments:
    model_name_or_path: str = None  # overwrite this
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    attn_implementation: str = None
    # --- fold PEFT args here ---
    lora: bool = False
    target_modules: str = "all-linear"
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    modules_to_save: str = None

    def __post_init__(self):
        self.model_name_or_path = resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class DataArguments:
    dataset_args: str = None  # overwrite this
    num_proc: int = 8
    disable_caching: bool = False
    max_length: int = 1024
    truncation: str = field(
        default="right",
        metadata={
            "help": (
                'The truncation strategy to use ("filter" or "right"). '
                '"filter" only keeps sequences that are shorter than max_length; '
                '"right" only keeps the rightmost max_length tokens for each sequence.'
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = None  # overwrite this
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    num_train_epochs: float = 4
    logging_steps: float = 10
    eval_on_start: bool = False
    eval_strategy: str = "steps"
    eval_steps: float = 0.25
    save_steps: float = 0.25
    save_only_model: bool = True
    # ===== REPA (Representation Alignment) options =====
    repa_enable: bool = False
    repa_teacher_name_or_path: str | None = None
    repa_teacher_pooling: str = "mean"  # "mean" | "cls" | "last_token"
    repa_target_layer: int = -1  # -1 for last_hidden_state
    repa_align_on: str = "clean"  # "clean" | "noised"
    repa_projection_type: str = "linear"  # "linear" | "mlp"
    repa_loss_type: str = "cosine"  # "cosine" | "mse"
    repa_loss_weight: float = 0.1
    repa_teacher_freeze: bool = True
    repa_temperature: float = 0.07
    repa_normalize: bool = False
    # Placement and precision for teacher encoder
    repa_teacher_device: str = "auto"  # "auto" | "student" | "cpu" | "cuda" | "cuda:0" | ...
    repa_teacher_dtype: str | None = None  # e.g., "bfloat16", "float16", "float32"
