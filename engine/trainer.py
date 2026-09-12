import gc
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm.auto import tqdm

from engine.models import load_base_model
from engine.dataset import create_dataloader, get_dataset_diagnostics
from engine.networks import (
    inject_network,
    inject_text_encoder_network,
    save_checkpoint,
    load_checkpoint,
    load_training_state,
)
from engine.sampler import generate_samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_save_dtype(precision_str):
    """Map a config save_precision string to a torch dtype."""
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32}.get(
        precision_str, torch.float16
    )


def _empty_device_cache():
    """Clear GPU memory cache — works on both CUDA (NVIDIA) and ROCm (AMD)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def _set_seed(seed: int, deterministic: bool = False):
    """Set global seeds for exact repeatability."""
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _create_optimizer(network_dict, config):
    """
    Build the optimizer respecting algorithm, separate learning rates for UNet & Text Encoders,
    UNet block weights (IN / MID / OUT), and weight decay.
    """
    unet_lr = config["learning_rate"]
    te_lr = config.get("text_encoder_lr", unet_lr)
    weight_decay = config.get("weight_decay", 0.01)

    down_weight = float(config.get("down_lr_weight", 1.0))
    mid_weight = float(config.get("mid_lr_weight", 1.0))
    up_weight = float(config.get("up_lr_weight", 1.0))

    # Build parameter groups
    param_groups = []

    def _add_unet_param_groups(unet_net):
        has_custom_blocks = (down_weight != 1.0 or mid_weight != 1.0 or up_weight != 1.0)
        if not has_custom_blocks:
            params = [p for p in unet_net.parameters() if p.requires_grad]
            if params:
                param_groups.append({"params": params, "lr": unet_lr})
            return

        down_params, mid_params, up_params = [], [], []
        for name, param in unet_net.named_parameters():
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            if "down" in name_lower:
                if down_weight <= 0.0:
                    param.requires_grad = False
                else:
                    down_params.append(param)
            elif "mid" in name_lower:
                if mid_weight <= 0.0:
                    param.requires_grad = False
                else:
                    mid_params.append(param)
            else:  # up blocks / other UNet layers
                if up_weight <= 0.0:
                    param.requires_grad = False
                else:
                    up_params.append(param)

        if down_params and down_weight > 0:
            param_groups.append({"params": down_params, "lr": unet_lr * down_weight})
        if mid_params and mid_weight > 0:
            param_groups.append({"params": mid_params, "lr": unet_lr * mid_weight})
        if up_params and up_weight > 0:
            param_groups.append({"params": up_params, "lr": unet_lr * up_weight})

    if isinstance(network_dict, dict):
        if "unet" in network_dict and network_dict["unet"] is not None:
            _add_unet_param_groups(network_dict["unet"])
        if "te" in network_dict and network_dict["te"] is not None:
            params = [p for p in network_dict["te"].parameters() if p.requires_grad]
            if params:
                param_groups.append({"params": params, "lr": te_lr})
        if "te1" in network_dict and network_dict["te1"] is not None:
            params = [p for p in network_dict["te1"].parameters() if p.requires_grad]
            if params:
                param_groups.append({"params": params, "lr": te_lr})
        if "te2" in network_dict and network_dict["te2"] is not None:
            params = [p for p in network_dict["te2"].parameters() if p.requires_grad]
            if params:
                param_groups.append({"params": params, "lr": te_lr})
    else:
        _add_unet_param_groups(network_dict)

    total_trainable_params = sum(sum(p.numel() for p in pg["params"]) for pg in param_groups)
    if not param_groups or total_trainable_params == 0:
        raise ValueError("Optimizer cannot be initialized: no trainable parameters found in adapter networks.")

    algo = config.get("optimizer", "AdamW")

    if algo == "AdamW":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif algo == "AdamW8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(param_groups, weight_decay=weight_decay)
        except ImportError:
            print("⚠️ bitsandbytes not installed — falling back to standard AdamW")
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif algo == "Prodigy":
        try:
            from prodigyopt import Prodigy
            d_coef = float(config.get("prodigy_d_coef", 1.0))
            use_bias_correction = bool(config.get("prodigy_use_bias_correction", False))
            safeguard_warmup = bool(config.get("prodigy_safeguard_warmup", False))
            decouple = bool(config.get("prodigy_decouple", True))
            for pg in param_groups:
                pg["lr"] = (pg["lr"] / unet_lr) if unet_lr > 0 else 1.0
            return Prodigy(
                param_groups,
                lr=1.0,
                weight_decay=weight_decay,
                d_coef=d_coef,
                use_bias_correction=use_bias_correction,
                safeguard_warmup=safeguard_warmup,
                decouple=decouple,
            )
        except ImportError:
            print("⚠️ prodigyopt not installed — falling back to standard AdamW")
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        print(f"⚠️ Unknown optimizer '{algo}' — falling back to AdamW")
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _create_scheduler(optimizer, scheduler_name, total_steps, warmup_steps=0, min_lr_ratio=0.0):
    """Create a learning-rate scheduler with optional warmup and min_lr_ratio."""
    safe_total_steps = max(1, total_steps)
    safe_warmup_steps = min(max(0, warmup_steps), safe_total_steps)

    if scheduler_name == "cosine":
        if safe_warmup_steps > 0:
            from transformers import get_cosine_schedule_with_warmup
            return get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=safe_warmup_steps, num_training_steps=safe_total_steps
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=safe_total_steps, eta_min=min_lr_ratio * optimizer.param_groups[0]["lr"]
        )
    elif scheduler_name == "cosine_with_warmup":
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=safe_warmup_steps, num_training_steps=safe_total_steps
        )
    elif scheduler_name == "constant_with_warmup":
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=safe_warmup_steps)
    elif scheduler_name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=max(0.0, min_lr_ratio), total_iters=safe_total_steps
        )
    elif scheduler_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=safe_total_steps)
    else:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=safe_total_steps)


def _compute_loss(noise_pred, target, timesteps, noise_scheduler, config):
    """Compute diffusion loss with optional Huber metric or Min-SNR gamma weighting."""
    loss_fn = config.get("loss_function", "mse")

    # Base loss
    if loss_fn == "huber":
        beta = config.get("huber_c", 0.1)
        loss = F.smooth_l1_loss(noise_pred.float(), target.float(), beta=beta, reduction="none")
    else:
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")

    # Min-SNR Gamma weighting (for DDPM models where alphas_cumprod is available)
    min_snr_gamma = config.get("min_snr_gamma", 0.0)
    if min_snr_gamma > 0 and hasattr(noise_scheduler, "alphas_cumprod"):
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
        alpha_t = alphas_cumprod[timesteps]
        snr = alpha_t / (1.0 - alpha_t)
        gamma = torch.full_like(snr, min_snr_gamma)

        # In v-prediction, denominator is increased by 1.0 (OneTrainer / Kohya standard)
        # User override via config takes precedence over scheduler metadata
        config_v_pred = config.get("v_prediction", False)
        scheduler_v_pred = getattr(noise_scheduler.config, "prediction_type", "epsilon") == "v_prediction"
        is_v_pred = config_v_pred or scheduler_v_pred
        snr_denominator = (snr + 1.0) if is_v_pred else snr
        snr_weight = torch.minimum(snr, gamma) / snr_denominator

        # Reshape to match loss dims
        while snr_weight.ndim < loss.ndim:
            snr_weight = snr_weight.unsqueeze(-1)
        loss = loss * snr_weight

    return loss.mean()


def _ddpm_add_noise(latents, noise, timesteps, alphas_cumprod):
    """
    Exact forward diffusion process q(x_t | x_0):
    x_t = sqrt(alpha_prod_t) * x_0 + sqrt(1 - alpha_prod_t) * noise
    Directly ported from OneTrainer's ModelSetupDiffusionMixin._add_noise_discrete.
    """
    sqrt_alpha = (alphas_cumprod[timesteps] ** 0.5).to(device=latents.device, dtype=latents.dtype)
    sqrt_one_minus_alpha = ((1.0 - alphas_cumprod[timesteps]) ** 0.5).to(device=latents.device, dtype=latents.dtype)
    while sqrt_alpha.ndim < latents.ndim:
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
    return sqrt_alpha * latents + sqrt_one_minus_alpha * noise


def _ddpm_get_velocity(latents, noise, timesteps, alphas_cumprod):
    """
    Exact v-prediction velocity target:
    v_t = sqrt(alpha_prod_t) * noise - sqrt(1 - alpha_prod_t) * latents
    Directly ported from OneTrainer / Salimans & Ho (2022).
    """
    sqrt_alpha = (alphas_cumprod[timesteps] ** 0.5).to(device=latents.device, dtype=latents.dtype)
    sqrt_one_minus_alpha = ((1.0 - alphas_cumprod[timesteps]) ** 0.5).to(device=latents.device, dtype=latents.dtype)
    while sqrt_alpha.ndim < latents.ndim:
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
    return sqrt_alpha * noise - sqrt_one_minus_alpha * latents


def _prune_old_checkpoints(backup_dir: Path, keep_last_n: int):
    """Deletes oldest checkpoint files if total count exceeds keep_last_n."""
    if keep_last_n <= 0:
        return
    ckpts = sorted(backup_dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    while len(ckpts) > keep_last_n:
        oldest = ckpts.pop(0)
        try:
            oldest.unlink()
            # Also remove corresponding .state.pt if present
            state_file = oldest.with_suffix(".state.pt")
            if state_file.exists():
                state_file.unlink()
            print(f"🗑️ Pruned old checkpoint: {oldest.name}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Flux-specific helpers
# ---------------------------------------------------------------------------

def _pack_flux_latents(latents, batch_size, channels, height, width):
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
    return latents


def _get_flux_ids(batch_size, seq_len, height, width, device, dtype):
    text_ids = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
    latent_h, latent_w = height // 2, width // 2
    latent_image_ids = torch.zeros(latent_h, latent_w, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = torch.arange(latent_h, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = torch.arange(latent_w, device=device, dtype=dtype)[None, :]
    latent_image_ids = latent_image_ids.reshape(1, -1, 3).expand(batch_size, -1, -1)
    return text_ids, latent_image_ids


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training_loop(config, run_event, stop_event, telemetry=None):
    """
    Core training entry point spawned by TrainingStateManager.
    Communicates live progress to the UI via the `telemetry` multiprocessing dict.
    """
    if telemetry is not None:
        telemetry["status"] = "PREPARING"

    # Reproducibility seed
    seed = config.get("training_seed", -1)
    _set_seed(seed, deterministic=config.get("deterministic", False))

    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=config.get("grad_accum_steps", 1),
    )
    accelerator.print(f"🚀 Engine started on device: {accelerator.device}")

    is_sdxl = config["model_type"] == "SDXL"
    is_flux = "Flux" in config["model_type"]
    is_dual_encoder = is_sdxl or is_flux
    train_te = config.get("text_encoder_training", False)
    cache_latents = config.get("cache_latents", True)
    save_dtype = _get_save_dtype(config.get("save_precision", "fp16"))
    grad_clip = config.get("gradient_clip_norm", 1.0)
    noise_offset_val = config.get("noise_offset", 0.0)

    # 1. Base Model Loading
    loaded_components = load_base_model(
        model_path=config["base_model_path"],
        model_type=config["model_type"],
        mixed_precision=config["mixed_precision"],
        quantization=config.get("quantization", "none"),
    )

    if is_dual_encoder:
        (tokenizer, tokenizer_2), (text_encoder, text_encoder_2), vae, unet, noise_scheduler = loaded_components
    else:
        tokenizer, text_encoder, vae, unet, noise_scheduler = loaded_components
        tokenizer_2 = text_encoder_2 = None

    # 2. Network Injection (UNet + Optional Text Encoders)
    unet_network = inject_network(
        base_model=unet,
        network_type=config["network_type"],
        rank=config["rank"],
        alpha=config["alpha"],
        conv_rank=config.get("conv_rank", 0),
        conv_alpha=config.get("conv_alpha", 0.0),
        network_dropout=config.get("network_dropout", 0.0),
        rank_dropout=config.get("rank_dropout", 0.0),
        module_dropout=config.get("module_dropout", 0.0),
        prefix="lora_unet",
    )

    networks_dict = {"unet": unet_network}

    if train_te:
        if is_dual_encoder:
            te1_network = inject_text_encoder_network(
                text_encoder=text_encoder,
                network_type=config["network_type"],
                rank=config.get("text_encoder_rank", config["rank"]),
                alpha=config.get("text_encoder_alpha", config["alpha"]),
                network_dropout=config.get("network_dropout", 0.0),
                rank_dropout=config.get("rank_dropout", 0.0),
                module_dropout=config.get("module_dropout", 0.0),
                prefix="lora_te1",
            )
            te2_network = inject_text_encoder_network(
                text_encoder=text_encoder_2,
                network_type=config["network_type"],
                rank=config.get("text_encoder_rank", config["rank"]),
                alpha=config.get("text_encoder_alpha", config["alpha"]),
                network_dropout=config.get("network_dropout", 0.0),
                rank_dropout=config.get("rank_dropout", 0.0),
                module_dropout=config.get("module_dropout", 0.0),
                prefix="lora_te2",
            )
            networks_dict["te1"] = te1_network
            networks_dict["te2"] = te2_network
        else:
            te_network = inject_text_encoder_network(
                text_encoder=text_encoder,
                network_type=config["network_type"],
                rank=config.get("text_encoder_rank", config["rank"]),
                alpha=config.get("text_encoder_alpha", config["alpha"]),
                network_dropout=config.get("network_dropout", 0.0),
                rank_dropout=config.get("rank_dropout", 0.0),
                module_dropout=config.get("module_dropout", 0.0),
                prefix="lora_te",
            )
            networks_dict["te"] = te_network

    # 2b. Gradient Checkpointing
    if config.get("gradient_checkpointing", True):
        if hasattr(unet, "enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
            accelerator.print("📉 Gradient checkpointing ON — saves VRAM at ~30% speed cost")
        else:
            accelerator.print("⚠️ Model does not support gradient checkpointing")

    # 3. Dataset Loading & Diagnostics
    accelerator.print("📂 Loading dataset and analyzing aspect ratios...")
    dataset, dataloader = create_dataloader(
        dataset_dir=config["dataset_dir"],
        batch_size=config["batch_size"],
        max_resolution=config["resolution"],
        repeats=config.get("dataset_repeats", 1),
        caption_extension=config.get("caption_extension", ".txt"),
        shuffle_captions=config.get("shuffle_captions", False),
        keep_tokens=config.get("keep_tokens", 0),
        caption_dropout_rate=config.get("caption_dropout_rate", 0.0),
        tag_dropout_rate=config.get("tag_dropout_rate", 0.0),
        horizontal_flip=config.get("horizontal_flip", False),
        color_augmentation=config.get("color_augmentation", False),
        random_crop=config.get("random_crop", False),
        min_bucket_res=config.get("min_bucket_resolution", 256),
        max_bucket_res=config.get("max_bucket_resolution", 2048),
        bucket_step=config.get("bucket_step", 64),
        num_workers=config.get("dataloader_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )

    num_epochs = config["epochs"]
    max_train_steps = config.get("max_train_steps", 0)
    diag = get_dataset_diagnostics(
        dataset=dataset,
        batch_size=config["batch_size"],
        grad_accum_steps=config.get("grad_accum_steps", 1),
        epochs=num_epochs,
        max_train_steps=max_train_steps,
    )

    if len(dataloader) == 0:
        raise RuntimeError(
            "Dataset and ARB sampler produced 0 batches. Ensure your dataset contains valid images "
            "and check batch size settings."
        )

    accelerator.print("\n" + "=" * 65)
    accelerator.print("📊 TRAINING RUN PRE-FLIGHT DIAGNOSTICS")
    accelerator.print("=" * 65)
    accelerator.print(f"  • Unique Images:                 {diag['unique_images']}")
    accelerator.print(f"  • Repeats:                       {diag['repeats']}")
    accelerator.print(f"  • Effective Samples:             {diag['effective_images']}")
    accelerator.print(f"  • Aspect Ratio Buckets Count:    {diag['buckets_count']}")
    accelerator.print(f"  • Batch Size:                    {diag['batch_size']}")
    accelerator.print(f"  • Gradient Accumulation Steps:   {diag['grad_accum_steps']}")
    accelerator.print(f"  • Batches per Epoch:             {diag['batches_per_epoch']}")
    accelerator.print(f"  • Optimizer Steps per Epoch:     {diag['optimizer_steps_per_epoch']}")
    accelerator.print(f"  • Total Epochs:                  {diag['epochs']}")
    accelerator.print(f"  • Expected Total Optimizer Steps: {diag['total_expected_optimizer_steps']}")
    accelerator.print("=" * 65 + "\n")

    # 3b. Caching
    # When text encoder training is enabled or dynamic captions are active, text caching is bypassed
    if cache_latents:
        dataset.cache_latents_and_embeds(
            vae=vae,
            tokenizer=tokenizer if not train_te else None,
            text_encoder=text_encoder if not train_te else None,
            device=accelerator.device,
            dtype=accelerator.unwrap_model(unet).dtype if hasattr(unet, 'dtype') else torch.float32,
            model_type=config["model_type"],
            tokenizer_2=tokenizer_2 if not train_te else None,
            text_encoder_2=text_encoder_2 if not train_te else None,
        )
        vae.cpu()
        has_cached_embeds = getattr(dataset, "cached_prompt_embeds", None) is not None
        if not train_te and has_cached_embeds:
            text_encoder.cpu()
            if text_encoder_2 is not None:
                text_encoder_2.cpu()
            accelerator.print("🧹 VAE and text encoders offloaded to CPU — GPU memory freed.")
        elif not train_te and not has_cached_embeds:
            text_encoder.to(accelerator.device)
            if text_encoder_2 is not None:
                text_encoder_2.to(accelerator.device)
            accelerator.print("🧹 VAE offloaded to CPU. Text Encoders active on GPU for dynamic caption encoding.")
        else:
            accelerator.print("🧹 VAE offloaded to CPU. Text Encoders active on GPU for LoRA training.")
        gc.collect()
        _empty_device_cache()

    # 4. Optimizer & Scheduler
    optimizer = _create_optimizer(networks_dict, config)

    total_steps = diag["total_expected_optimizer_steps"]
    warmup_steps = int(config.get("warmup_ratio", 0.0) * total_steps)
    if config.get("warmup_steps", 0) > 0:
        warmup_steps = config["warmup_steps"]

    scheduler = _create_scheduler(
        optimizer,
        config.get("lr_scheduler", "cosine"),
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=config.get("min_lr_ratio", 0.0),
    )

    # 4b. Resume from checkpoint / training state if specified
    resume_path = config.get("resume_from_checkpoint", "")
    start_epoch = 0
    global_step = 0
    if resume_path and Path(resume_path).exists():
        accelerator.print(f"🔄 Resuming adapter weights from checkpoint: {resume_path}")
        load_checkpoint(networks_dict if train_te else unet_network, resume_path)
        restored = load_training_state(resume_path, optimizer=optimizer, scheduler=scheduler)
        if restored:
            start_epoch = restored.get("epoch", 0)
            global_step = restored.get("global_step", 0)
            accelerator.print(f"✅ Restored training state: starting from Epoch {start_epoch + 1}, Step {global_step}")

    # 5. Prepare with Accelerator
    prepared_items = [unet, unet_network]
    if train_te:
        if is_dual_encoder:
            prepared_items.extend([te1_network, te2_network])
        else:
            prepared_items.append(te_network)
    prepared_items.extend([optimizer, dataloader, scheduler])

    prep_results = accelerator.prepare(*prepared_items)
    unet = prep_results[0]
    unet_network = prep_results[1]
    idx = 2
    if train_te:
        if is_dual_encoder:
            te1_network = prep_results[idx]
            te2_network = prep_results[idx + 1]
            idx += 2
        else:
            te_network = prep_results[idx]
            idx += 1
    optimizer = prep_results[idx]
    dataloader = prep_results[idx + 1]
    scheduler = prep_results[idx + 2]

    if not cache_latents:
        vae.to(accelerator.device)
    if train_te or not cache_latents:
        text_encoder.to(accelerator.device)
        if text_encoder_2 is not None:
            text_encoder_2.to(accelerator.device)

    # 6. Directories
    backup_dir = Path(config.get("output_dir", "checkpoints/final")).parent / "backups"
    final_dir = Path(config.get("output_dir", "checkpoints/final"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # 7. Training Loop Setup
    accelerator.print("🔥 Starting Training Loop...")
    if telemetry is not None:
        telemetry["status"] = "TRAINING"
        telemetry["total_epochs"] = num_epochs
        telemetry["total_steps"] = total_steps

    start_time = time.time()
    save_training_state_flag = config.get("save_training_state", True)

    # TensorBoard logging
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = Path(config.get("output_dir", "checkpoints/final")) / "logs"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
    except Exception as e:
        accelerator.print(f"⚠️ TensorBoard writer disabled: {e}")

    progress_bar = tqdm(
        total=total_steps,
        initial=global_step,
        desc="Training Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(start_epoch, num_epochs):
        unet.train()
        unet_network.train()
        if train_te:
            if is_dual_encoder:
                te1_network.train()
                te2_network.train()
            else:
                te_network.train()

        if telemetry is not None:
            telemetry["epoch"] = epoch + 1

        for batch in dataloader:
            step_start = time.time()

            # Signals
            if stop_event.is_set():
                accelerator.print(f"\n🛑 Stop signal received at step {global_step}. Saving and exiting...")
                break
            run_event.wait()

            with accelerator.accumulate(unet):
                # A. Latents
                if cache_latents:
                    latents = batch["latents"].to(accelerator.device)
                else:
                    with torch.no_grad():
                        pixel_values = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        if is_flux:
                            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                        else:
                            latents = latents * vae.config.scaling_factor

                bsz, channels, height, width = latents.shape
                noise = torch.randn_like(latents)

                # Noise offset
                if noise_offset_val > 0:
                    noise = noise + noise_offset_val * torch.randn(
                        (bsz, channels, 1, 1), device=latents.device, dtype=latents.dtype
                    )

                # B. Text embeddings
                if cache_latents and not train_te and "prompt_embeds" in batch:
                    prompt_embeds = batch["prompt_embeds"].to(accelerator.device)
                    if is_dual_encoder:
                        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(accelerator.device)
                else:
                    # Compute live text embeddings with gradients if TE training is active
                    context_manager = torch.enable_grad() if train_te else torch.no_grad()
                    with context_manager:
                        if is_dual_encoder:
                            inp_1 = tokenizer(
                                batch["captions"], padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt",
                            ).input_ids.to(accelerator.device)

                            inp_2 = tokenizer_2(
                                batch["captions"], padding="max_length",
                                max_length=tokenizer_2.model_max_length,
                                truncation=True, return_tensors="pt",
                            ).input_ids.to(accelerator.device)

                            out_1 = text_encoder(inp_1, output_hidden_states=True)
                            out_2 = text_encoder_2(inp_2, output_hidden_states=True)

                            if is_flux:
                                pooled_prompt_embeds = out_1.pooler_output
                                prompt_embeds = out_2.last_hidden_state
                            else:
                                prompt_embeds = torch.cat(
                                    [out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1
                                )
                                pooled_prompt_embeds = out_2.text_embeds
                        else:
                            inputs = tokenizer(
                                batch["captions"], padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt",
                            )
                            prompt_embeds = text_encoder(inputs.input_ids.to(accelerator.device))[0]

                # C. Forward pass & Dynamic Target Calculation
                if is_flux:
                    packed_latents = _pack_flux_latents(latents, bsz, channels, height, width)
                    packed_noise = _pack_flux_latents(noise, bsz, channels, height, width)
                    txt_ids, img_ids = _get_flux_ids(
                        bsz, prompt_embeds.shape[1], height, width,
                        accelerator.device, prompt_embeds.dtype,
                    )

                    u = torch.rand((bsz, 1, 1), device=latents.device, dtype=prompt_embeds.dtype)
                    noisy_latents = (1.0 - u) * packed_latents + u * packed_noise
                    target = packed_noise - packed_latents
                    num_timesteps = getattr(noise_scheduler.config, "num_train_timesteps", 1000)
                    timesteps = (u.flatten() * num_timesteps).long()

                    noise_pred = unet(
                        hidden_states=noisy_latents,
                        timestep=timesteps / 1000,
                        guidance=torch.full((bsz,), 4.0, device=latents.device, dtype=prompt_embeds.dtype),
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=txt_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0]

                else:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (bsz,), device=latents.device,
                    ).long()

                    # Exact forward diffusion noise addition ported from OneTrainer
                    if hasattr(noise_scheduler, "alphas_cumprod"):
                        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=latents.device, dtype=torch.float32)
                        noisy_latents = _ddpm_add_noise(latents, noise, timesteps, alphas_cumprod)
                    else:
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                        alphas_cumprod = None

                    # Dynamic target calculation for epsilon vs v_prediction
                    # User override via config takes precedence over scheduler metadata
                    config_v_pred = config.get("v_prediction", False)
                    scheduler_v_pred = getattr(noise_scheduler.config, "prediction_type", "epsilon") == "v_prediction"
                    is_v_pred = config_v_pred or scheduler_v_pred
                    if is_v_pred:
                        if alphas_cumprod is not None:
                            target = _ddpm_get_velocity(latents, noise, timesteps, alphas_cumprod)
                        else:
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    if is_sdxl:
                        pixel_h = height * 8  # latent→pixel
                        pixel_w = width * 8
                        add_time_ids = torch.tensor(
                            [[pixel_h, pixel_w, 0, 0, pixel_h, pixel_w]],
                            dtype=prompt_embeds.dtype,
                        ).to(accelerator.device).repeat(bsz, 1)
                        added_cond_kwargs = {
                            "text_embeds": pooled_prompt_embeds,
                            "time_ids": add_time_ids,
                        }
                        noise_pred = unet(
                            noisy_latents, timesteps, prompt_embeds,
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample
                    else:
                        noise_pred = unet(noisy_latents, timesteps, prompt_embeds).sample

                # D. Loss calculation with Huber / Min-SNR
                loss = _compute_loss(noise_pred, target, timesteps, noise_scheduler, config)
                accelerator.backward(loss)

                if grad_clip > 0:
                    params_to_clip = list(unet_network.parameters())
                    if train_te:
                        if is_dual_encoder:
                            params_to_clip.extend(list(te1_network.parameters()) + list(te2_network.parameters()))
                        else:
                            params_to_clip.extend(list(te_network.parameters()))
                    accelerator.clip_grad_norm_(params_to_clip, grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                # Performance telemetry
                step_duration = max(0.001, time.time() - step_start)
                steps_per_sec = 1.0 / step_duration
                elapsed = time.time() - start_time
                remaining_steps = max(0, total_steps - global_step)
                eta_seconds = remaining_steps / (global_step / max(1.0, elapsed)) if global_step > 0 else 0

                current_lr = scheduler.get_last_lr()[0]
                loss_val = float(loss.item())

                # Update Telemetry Dict
                if telemetry is not None:
                    telemetry["step"] = global_step
                    telemetry["loss"] = loss_val
                    telemetry["lr"] = current_lr
                    telemetry["steps_per_sec"] = steps_per_sec
                    telemetry["eta_seconds"] = eta_seconds

                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", loss_val, global_step)
                    tb_writer.add_scalar("train/lr", current_lr, global_step)
                    tb_writer.add_scalar("train/steps_per_sec", steps_per_sec, global_step)
                    tb_writer.add_scalar("train/epoch", epoch + 1, global_step)

                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "epoch": f"{epoch + 1}/{num_epochs}",
                })

                # Checkpoint backup by steps
                save_steps = config.get("save_every_n_steps", 100)
                if save_steps > 0 and global_step % save_steps == 0:
                    accelerator.print(f"\n💾 Step {global_step} | Auto-saving checkpoint...")
                    unwrapped_unet = accelerator.unwrap_model(unet_network)
                    save_dict = {"unet": unwrapped_unet}
                    if train_te:
                        if is_dual_encoder:
                            save_dict["te1"] = accelerator.unwrap_model(te1_network)
                            save_dict["te2"] = accelerator.unwrap_model(te2_network)
                        else:
                            save_dict["te"] = accelerator.unwrap_model(te_network)

                    ckpt_path = backup_dir / f"step_{global_step}.safetensors"
                    save_checkpoint(
                        network=save_dict,
                        save_path=ckpt_path,
                        dtype=save_dtype,
                        config=config,
                        global_step=global_step,
                        epoch=epoch + 1,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        save_state=save_training_state_flag,
                    )
                    _prune_old_checkpoints(backup_dir, config.get("keep_last_n_checkpoints", 0))

                # Sample Generation by steps
                sample_steps_interval = config.get("sample_every_n_steps", 0)
                if sample_steps_interval > 0 and global_step % sample_steps_interval == 0:
                    generate_samples(
                        unet=accelerator.unwrap_model(unet),
                        vae=vae,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        config=config,
                        current_step=global_step,
                        current_epoch=epoch + 1,
                        tokenizer_2=tokenizer_2,
                        text_encoder_2=text_encoder_2,
                        device=accelerator.device,
                    )

                if max_train_steps > 0 and global_step >= max_train_steps:
                    break

        # End of Epoch actions
        save_epochs = config.get("save_every_n_epochs", 0)
        if save_epochs > 0 and (epoch + 1) % save_epochs == 0:
            accelerator.print(f"\n💾 Epoch {epoch + 1} | Saving epoch checkpoint...")
            unwrapped_unet = accelerator.unwrap_model(unet_network)
            save_dict = {"unet": unwrapped_unet}
            if train_te:
                if is_dual_encoder:
                    save_dict["te1"] = accelerator.unwrap_model(te1_network)
                    save_dict["te2"] = accelerator.unwrap_model(te2_network)
                else:
                    save_dict["te"] = accelerator.unwrap_model(te_network)

            ckpt_path = backup_dir / f"epoch_{epoch + 1}.safetensors"
            save_checkpoint(
                network=save_dict,
                save_path=ckpt_path,
                dtype=save_dtype,
                config=config,
                global_step=global_step,
                epoch=epoch + 1,
                optimizer=optimizer,
                scheduler=scheduler,
                save_state=save_training_state_flag,
            )
            _prune_old_checkpoints(backup_dir, config.get("keep_last_n_checkpoints", 0))

        # Sample Generation by epoch
        sample_epochs_interval = config.get("sample_every_n_epochs", 0)
        if sample_epochs_interval > 0 and (epoch + 1) % sample_epochs_interval == 0:
            generate_samples(
                unet=accelerator.unwrap_model(unet),
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                config=config,
                current_step=global_step,
                current_epoch=epoch + 1,
                tokenizer_2=tokenizer_2,
                text_encoder_2=text_encoder_2,
                device=accelerator.device,
            )

        if stop_event.is_set():
            break
        if max_train_steps > 0 and global_step >= max_train_steps:
            break

    progress_bar.close()

    # 8. Final Model Saving
    if not stop_event.is_set():
        accelerator.print(f"\n🎉 Training Complete! Saving final weights to {final_dir}...")
    else:
        accelerator.print(f"\n💾 Training stopped. Saving final weights to {final_dir}...")

    unwrapped_unet = accelerator.unwrap_model(unet_network)
    save_dict = {"unet": unwrapped_unet}
    if train_te:
        if is_dual_encoder:
            save_dict["te1"] = accelerator.unwrap_model(te1_network)
            save_dict["te2"] = accelerator.unwrap_model(te2_network)
        else:
            save_dict["te"] = accelerator.unwrap_model(te_network)

    proj_name = config.get("project_name", "my_lora").replace(".safetensors", "")
    final_path = final_dir / f"{proj_name}.safetensors"
    final_epoch = num_epochs if not stop_event.is_set() else (epoch + 1 if 'epoch' in locals() else 0)
    save_checkpoint(
        network=save_dict,
        save_path=final_path,
        dtype=save_dtype,
        config=config,
        global_step=global_step,
        epoch=final_epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        save_state=save_training_state_flag,
    )

    accelerator.print(f"✅ Saved: {final_path}")

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()