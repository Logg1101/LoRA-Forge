"""
Typed training configuration using Pydantic v2.
Every UI control maps to a field here. Serializable to/from JSON.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class TrainingConfig(BaseModel):
    """Complete training configuration for LoRA Forge."""

    # ── Project ──────────────────────────────────────────────────────
    project_name: str = "my_first_lora"
    base_model_path: str = ""
    model_type: str = "SDXL"  # SD1.5, SDXL, Flux.1, Flux Schnell, Flux Dev
    dataset_dir: str = ""
    output_dir: str = "checkpoints/final"

    # ── Dataset ──────────────────────────────────────────────────────
    dataset_repeats: int = Field(default=1, ge=1, le=100)
    caption_extension: str = ".txt"
    shuffle_captions: bool = False
    keep_tokens: int = Field(default=0, ge=0, le=100)
    caption_dropout_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    tag_dropout_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    min_bucket_resolution: int = Field(default=256, ge=64)
    max_bucket_resolution: int = Field(default=2048, le=4096)
    bucket_step: int = Field(default=64, ge=8)

    # ── Augmentation ─────────────────────────────────────────────────
    horizontal_flip: bool = False
    color_augmentation: bool = False
    random_crop: bool = False

    # ── Network ──────────────────────────────────────────────────────
    network_type: str = "lora"  # lora, dora, loha, locon
    rank: int = Field(default=16, ge=1, le=512)
    alpha: float = Field(default=8.0, ge=0.01)
    network_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    rank_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    module_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    conv_rank: int = Field(default=0, ge=0, le=512)
    conv_alpha: float = Field(default=0.0, ge=0.0)
    train_conv: bool = False
    down_lr_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    mid_lr_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    up_lr_weight: float = Field(default=1.0, ge=0.0, le=10.0)

    # ── Training ─────────────────────────────────────────────────────
    batch_size: int = Field(default=1, ge=1, le=64)
    epochs: int = Field(default=10, ge=1, le=10000)
    max_train_steps: int = Field(default=0, ge=0)
    grad_accum_steps: int = Field(default=1, ge=1, le=128)
    gradient_clip_norm: float = Field(default=1.0, ge=0.0)
    learning_rate: float = Field(default=1e-4, gt=0)
    text_encoder_lr: float = Field(default=5e-5, gt=0)
    text_encoder_training: bool = False
    optimizer: str = "AdamW8bit"  # AdamW, AdamW8bit, Prodigy
    prodigy_d_coef: float = Field(default=1.0, gt=0.0)
    prodigy_use_bias_correction: bool = False
    prodigy_safeguard_warmup: bool = False
    prodigy_decouple: bool = True
    lr_scheduler: str = "cosine"  # cosine, constant, linear, cosine_with_warmup, constant_with_warmup
    warmup_steps: int = Field(default=0, ge=0)
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    min_lr_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)

    # ── Memory / Performance ─────────────────────────────────────────
    resolution: int = Field(default=1024, ge=64)
    mixed_precision: str = "bf16"  # bf16, fp16, fp8, no
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_text_encoder_outputs: bool = True
    quantization: str = "none"  # none, nf4, int8
    dataloader_workers: int = Field(default=2, ge=0, le=16)
    persistent_workers: bool = False
    pin_memory: bool = True
    prefetch_factor: int = Field(default=2, ge=1, le=16)

    # ── Loss / Diffusion ─────────────────────────────────────────────
    loss_function: str = "mse"  # mse, huber
    huber_c: float = Field(default=0.1, gt=0)
    min_snr_gamma: float = Field(default=0.0, ge=0.0)
    noise_offset: float = Field(default=0.0, ge=0.0, le=1.0)
    v_prediction: bool = False

    # ── VAE ───────────────────────────────────────────────────────────
    use_custom_vae: bool = False
    custom_vae_path: str = ""
    vae_precision: str = "fp32"  # fp32, fp16, bf16

    # ── Seed / Reproducibility ───────────────────────────────────────
    training_seed: int = Field(default=-1)  # -1 = random
    shuffle_seed: int = Field(default=-1)   # -1 = random
    deterministic: bool = False

    # ── Sample Generation ────────────────────────────────────────────
    enable_samples: bool = False
    sample_every_n_epochs: int = Field(default=0, ge=0)
    sample_every_n_steps: int = Field(default=0, ge=0)
    sample_prompts: List[str] = Field(default_factory=lambda: [
        "portrait, cinematic lighting, detailed",
    ])
    sample_negative_prompt: str = ""
    sample_sampler: str = "euler"
    sample_steps: int = Field(default=28, ge=1, le=150)
    sample_cfg: float = Field(default=7.0, ge=1.0, le=30.0)
    sample_resolution: int = Field(default=512, ge=64)
    sample_seed: int = Field(default=42)
    sample_output_dir: str = "samples"

    # ── Checkpoints ──────────────────────────────────────────────────
    save_every_n_steps: int = Field(default=100, ge=0)
    save_every_n_epochs: int = Field(default=0, ge=0)
    save_precision: str = "fp16"  # fp16, bf16, float32
    keep_last_n_checkpoints: int = Field(default=0, ge=0)  # 0 = keep all
    save_optimizer_state: bool = False
    save_training_state: bool = False
    resume_from_checkpoint: str = ""

    # ── Validators ───────────────────────────────────────────────────
    @field_validator("resolution", "sample_resolution", "min_bucket_resolution", "max_bucket_resolution")
    @classmethod
    def resolution_divisible_by_8(cls, v: int) -> int:
        if v % 8 != 0:
            v = (v // 8) * 8
        return max(v, 64)

    # ── Computed Properties ──────────────────────────────────────────
    @property
    def is_flux(self) -> bool:
        return "Flux" in self.model_type

    @property
    def is_sdxl(self) -> bool:
        return self.model_type == "SDXL"

    @property
    def is_dual_encoder(self) -> bool:
        return self.is_sdxl or self.is_flux

    def training_plan(self, image_count: int = 0) -> dict:
        """Compute the training plan summary."""
        effective_images = image_count * self.dataset_repeats if image_count > 0 else 0
        batches_per_epoch = math.ceil(effective_images / self.batch_size) if effective_images > 0 else 0
        steps_per_epoch = math.ceil(batches_per_epoch / max(1, self.grad_accum_steps)) if batches_per_epoch > 0 else 0
        total_steps = (
            self.max_train_steps
            if self.max_train_steps > 0
            else self.epochs * steps_per_epoch
        )
        return {
            "image_count": image_count,
            "repeats": self.dataset_repeats,
            "effective_images": effective_images,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "total_epochs": self.epochs,
            "batch_size": self.batch_size,
            "resolution": self.resolution,
            "network_type": self.network_type,
            "rank": self.rank,
            "alpha": self.alpha,
            "unet_lr": self.learning_rate,
            "te_lr": self.text_encoder_lr,
            "optimizer": self.optimizer,
            "scheduler": self.lr_scheduler,
        }

    def validate_for_training(self) -> List[str]:
        """Return a list of fatal errors that prevent training from starting."""
        errors = []
        if not self.base_model_path or not self.base_model_path.strip():
            errors.append("Base model path is not specified.")
        elif not Path(self.base_model_path).exists():
            errors.append(f"Base model path does not exist: {self.base_model_path}")

        if not self.dataset_dir or not self.dataset_dir.strip():
            errors.append("Dataset directory is not specified.")
        elif not Path(self.dataset_dir).exists():
            errors.append(f"Dataset directory does not exist: {self.dataset_dir}")

        if self.batch_size < 1:
            errors.append(f"Batch size must be >= 1 (got {self.batch_size}).")
        if self.epochs < 1:
            errors.append(f"Epochs must be >= 1 (got {self.epochs}).")
        if self.rank < 1:
            errors.append(f"Adapter rank must be >= 1 (got {self.rank}).")
        if self.learning_rate <= 0:
            errors.append(f"Learning rate must be > 0 (got {self.learning_rate}).")

        valid_networks = {"lora", "dora", "loha", "locon", "lycoris"}
        if self.network_type.lower() not in valid_networks:
            errors.append(f"Network type '{self.network_type}' is not supported. Must be one of {valid_networks}.")

        return errors

    def validate_config(self) -> List[str]:
        """Return a list of advisory warnings about potentially sub-optimal settings."""
        warnings = []
        if self.text_encoder_training and self.cache_latents:
            warnings.append("Text encoder training with 'Cache Latents' will compute text embeddings live on GPU while caching image latents.")
        if self.quantization != "none" and self.model_type == "SD1.5":
            warnings.append("Quantization has minimal benefit for SD1.5 — it is already lightweight.")
        if self.is_flux and self.resolution > 768 and self.quantization == "none":
            warnings.append("Flux at >768 resolution without quantization may exceed 12GB VRAM.")
        if self.noise_offset > 0 and self.v_prediction:
            warnings.append("Noise offset with v-prediction may produce unexpected results.")
        if self.conv_rank > 0 and self.network_type not in ("locon", "loha"):
            warnings.append(f"Conv rank ({self.conv_rank}) is only used by LoCon/LoHA, ignored by {self.network_type}.")
        return warnings

    def to_engine_dict(self) -> dict:
        """Convert to the flat dict format the training engine expects."""
        return self.model_dump()

    @classmethod
    def from_legacy_dict(cls, d: dict) -> "TrainingConfig":
        """Load from legacy configs/default.json format, ignoring unknown keys."""
        known_fields = set(cls.model_fields.keys())
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)
