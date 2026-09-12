import inspect
import json
import os
from pathlib import Path
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from lycoris import create_lycoris
    from lycoris.kohya import LycorisNetwork
except ImportError:
    raise ImportError("lycoris is not installed. Install it with: pip install lycoris-lora")


def inject_network(
    base_model,
    network_type="lora",
    rank=16,
    alpha=8.0,
    multiplier=1.0,
    conv_rank=0,
    conv_alpha=0.0,
    network_dropout=0.0,
    rank_dropout=0.0,
    module_dropout=0.0,
    prefix="lora_unet",
):
    """
    Injects trainable adapter layers (LoRA, DoRA, LoHA, LoCon, etc.) into a frozen base model.

    After injection, the forward pass still goes through `base_model(...)`.
    The LyCORIS wrapper manages the adapter weights for saving/loading.
    """
    algo = network_type.lower().strip()
    print(f"🧬 Injecting {algo.upper()} layers into {prefix} (Rank: {rank}, Alpha: {alpha})...")

    supported_algos = ["lora", "dora", "loha", "locon", "lycoris"]
    if algo not in supported_algos:
        raise ValueError(f"❌ Unsupported network type '{algo}'. Must be one of {supported_algos}.")

    # Configure prefix for Kohya / Diffusers compatibility
    try:
        LycorisNetwork.LORA_PREFIX = prefix
    except Exception:
        pass

    # Build kwargs for LyCORIS
    kwargs = {
        "multiplier": multiplier,
        "linear_dim": rank,
        "linear_alpha": alpha,
        "algo": algo,
    }

    # Conv dimensions for LoCon / LoHA
    if conv_rank > 0:
        kwargs["conv_dim"] = conv_rank
        kwargs["conv_alpha"] = conv_alpha if conv_alpha > 0 else (conv_rank / 2.0)

    # Dropout parameters
    if network_dropout > 0:
        kwargs["dropout"] = network_dropout
    if rank_dropout > 0:
        kwargs["rank_dropout"] = rank_dropout
    if module_dropout > 0:
        kwargs["module_dropout"] = module_dropout

    # Create adapter
    network = create_lycoris(base_model, **kwargs)
    network.apply_to()

    # Freeze base model, unfreeze network
    base_model.requires_grad_(False)
    network.requires_grad_(True)

    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    if trainable_params == 0:
        raise RuntimeError(
            f"❌ Adapter injection failed for {prefix}: 0 trainable parameters created. "
            f"Verify network_type '{network_type}', rank {rank}, and base model layer compatibility."
        )
    total_params = sum(p.numel() for p in base_model.parameters())
    ratio = trainable_params / total_params * 100 if total_params > 0 else 0
    print(f"✅ Injection complete for {prefix}. Trainable: {trainable_params:,} / {total_params:,} params ({ratio:.2f}%)")

    return network


def inject_text_encoder_network(
    text_encoder,
    network_type="lora",
    rank=16,
    alpha=8.0,
    multiplier=1.0,
    network_dropout=0.0,
    rank_dropout=0.0,
    module_dropout=0.0,
    prefix="lora_te1",
):
    """Injects adapter layers into a CLIP text encoder."""
    return inject_network(
        base_model=text_encoder,
        network_type=network_type,
        rank=rank,
        alpha=alpha,
        multiplier=multiplier,
        conv_rank=0,
        conv_alpha=0.0,
        network_dropout=network_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        prefix=prefix,
    )


def build_metadata(
    config: dict | None = None,
    global_step: int | None = None,
    epoch: int | None = None,
    extra_metadata: dict | None = None,
) -> dict[str, str]:
    """
    Constructs a validated, string-only metadata dictionary for .safetensors headers.
    Compatible with standard Kohya, Civitai, ComfyUI, and SD-WebUI metadata conventions.
    """
    meta: dict[str, str] = {}
    config = config or {}

    # Core module identification
    meta["ss_network_module"] = "lycoris.kohya"

    # Adapter type & architecture
    if "network_type" in config:
        meta["ss_network_type"] = str(config["network_type"])
        meta["modelspec.encoder"] = str(config["network_type"])

    if "rank" in config:
        meta["ss_network_dim"] = str(config["rank"])
    if "alpha" in config:
        meta["ss_network_alpha"] = str(config["alpha"])

    # Detailed network arguments as JSON string
    net_args = {}
    for k in (
        "network_type",
        "conv_rank",
        "conv_alpha",
        "network_dropout",
        "rank_dropout",
        "module_dropout",
    ):
        if k in config:
            net_args[k] = config[k]
    if net_args:
        meta["ss_network_args"] = json.dumps(net_args)

    # Base model information (OneTrainer / Kohya / WebUI standard conventions)
    if "model_type" in config:
        m_type = str(config["model_type"]).upper()
        if m_type == "SDXL":
            meta["ss_base_model_version"] = "sdxl_base_v1-0"
            meta["modelspec.architecture"] = "stable-diffusion-xl-v1-0"
        elif m_type == "SD1.5":
            meta["ss_base_model_version"] = "sd_v1-5"
            meta["modelspec.architecture"] = "stable-diffusion-v1-5"
        elif "FLUX" in m_type:
            meta["ss_base_model_version"] = "flux"
            meta["modelspec.architecture"] = "flux-1"
        else:
            meta["ss_base_model_version"] = str(config["model_type"])
            meta["modelspec.architecture"] = str(config["model_type"])

    # Prediction type metadata
    if config.get("v_prediction", False):
        meta["ss_v_prediction"] = "True"
        meta["modelspec.prediction_type"] = "v"
    else:
        meta["ss_v_prediction"] = "False"
        meta["modelspec.prediction_type"] = "epsilon"

    if "base_model_path" in config and config["base_model_path"]:
        meta["ss_sd_model_name"] = Path(config["base_model_path"]).name

    if "project_name" in config and config["project_name"]:
        meta["modelspec.title"] = str(config["project_name"])

    # Hyperparameters
    if "learning_rate" in config:
        meta["ss_learning_rate"] = str(config["learning_rate"])
        meta["ss_unet_lr"] = str(config["learning_rate"])
    if "text_encoder_lr" in config:
        meta["ss_text_encoder_lr"] = str(config["text_encoder_lr"])
    if "optimizer" in config:
        meta["ss_optimizer"] = str(config["optimizer"])
    if "lr_scheduler" in config:
        meta["ss_lr_scheduler"] = str(config["lr_scheduler"])
    if "resolution" in config:
        res = config["resolution"]
        meta["ss_resolution"] = f"({res}, {res})"
    if "batch_size" in config:
        meta["ss_batch_size_per_device"] = str(config["batch_size"])
    if "grad_accum_steps" in config:
        meta["ss_gradient_accumulation_steps"] = str(config["grad_accum_steps"])
    if "mixed_precision" in config:
        meta["ss_mixed_precision"] = str(config["mixed_precision"])
    if "gradient_checkpointing" in config:
        meta["ss_gradient_checkpointing"] = str(config["gradient_checkpointing"])
    if "loss_function" in config:
        meta["ss_loss_function"] = str(config["loss_function"])
    if "noise_offset" in config:
        meta["ss_noise_offset"] = str(config["noise_offset"])
    if "min_snr_gamma" in config:
        meta["ss_min_snr_gamma"] = str(config["min_snr_gamma"])

    # Progress tracking
    if global_step is not None:
        meta["ss_steps"] = str(global_step)
    if epoch is not None:
        meta["ss_epoch"] = str(epoch)
    if "max_train_steps" in config:
        meta["ss_max_train_steps"] = str(config["max_train_steps"])
    if "epochs" in config:
        meta["ss_total_epochs"] = str(config["epochs"])

    meta["ss_training_comment"] = "Trained with LoRA Forge"

    # Merge custom/extra metadata if provided
    if extra_metadata:
        for k, v in extra_metadata.items():
            if isinstance(v, (dict, list)):
                meta[str(k)] = json.dumps(v)
            else:
                meta[str(k)] = str(v)

    # Strictly enforce that all keys and values are str
    return {str(k): str(v) for k, v in meta.items()}


def _extract_standard_state_dict(network, dtype=torch.float16, default_prefix="lora_unet_"):
    """Extracts state dict from a network or dict, enforcing standard prefix."""
    sd = network.state_dict() if hasattr(network, "state_dict") else network
    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().clone().to("cpu")
            if dtype is not None:
                v = v.to(dtype)
        # Normalize prefix
        if k.startswith("lycoris_"):
            std_k = k.replace("lycoris_", default_prefix, 1)
        elif not (k.startswith("lora_unet_") or k.startswith("lora_te") or k.startswith("lora_te1_") or k.startswith("lora_te2_")):
            std_k = f"{default_prefix}{k}"
        else:
            std_k = k
        out[std_k] = v
    return out


def save_checkpoint(
    network,
    save_path: str | Path,
    dtype: torch.dtype = torch.float16,
    config: dict | None = None,
    global_step: int | None = None,
    epoch: int | None = None,
    extra_metadata: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    save_state: bool = False,
) -> Path:
    """
    Universal checkpoint serialization function for LoRA / LyCORIS models.
    Supports single or multi-network adapters (UNet + Text Encoders).
    Optionally saves optimizer & scheduler state for seamless resuming.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(
        config=config,
        global_step=global_step,
        epoch=epoch,
        extra_metadata=extra_metadata,
    )

    combined_state_dict = {}

    if isinstance(network, dict):
        for part_name, net in network.items():
            if net is not None:
                default_prefix = f"lora_{part_name}_" if not part_name.startswith("lora_") else f"{part_name}_"
                if part_name in ("te", "te1"):
                    default_prefix = "lora_te1_"
                elif part_name == "te2":
                    default_prefix = "lora_te2_"
                elif part_name == "unet":
                    default_prefix = "lora_unet_"
                part_sd = _extract_standard_state_dict(net, dtype=dtype, default_prefix=default_prefix)
                combined_state_dict.update(part_sd)
    elif isinstance(network, (list, tuple)):
        for net in network:
            if net is not None:
                combined_state_dict.update(_extract_standard_state_dict(net, dtype=dtype))
    else:
        combined_state_dict = _extract_standard_state_dict(network, dtype=dtype)

    if save_path.suffix.lower() == ".safetensors":
        from safetensors.torch import save_file
        save_file(combined_state_dict, str(save_path), metadata=metadata)
    else:
        torch.save(combined_state_dict, str(save_path))

    # Save training state if requested
    if save_state and optimizer is not None:
        state_file = save_path.with_suffix(".state.pt")
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
        }
        torch.save(training_state, str(state_file))

    return save_path


def load_checkpoint(network, checkpoint_path: str | Path) -> dict:
    """
    Universal checkpoint loader for resuming or fine-tuning from existing adapter weights.
    Supports single or multi-network adapters seamlessly.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if checkpoint_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(checkpoint_path))
    else:
        sd = torch.load(str(checkpoint_path), map_location="cpu")

    results = {}

    if isinstance(network, dict):
        for part_name, net in network.items():
            if net is not None:
                part_prefix = f"lora_{part_name}_" if not part_name.startswith("lora_") else f"{part_name}_"
                if part_name in ("te", "te1"):
                    part_prefix = "lora_te1_"
                elif part_name == "te2":
                    part_prefix = "lora_te2_"
                elif part_name == "unet":
                    part_prefix = "lora_unet_"

                part_sd = {
                    k.replace(part_prefix, "lycoris_", 1) if k.startswith(part_prefix) else k: v
                    for k, v in sd.items()
                    if k.startswith(part_prefix) or (part_name == "unet" and k.startswith("lora_unet_"))
                }
                if hasattr(net, "load_state_dict"):
                    results[part_name] = net.load_state_dict(part_sd, strict=False)
        return results

    # Single network load
    net_sd = network.state_dict() if hasattr(network, "state_dict") else {}
    net_has_lycoris = any(k.startswith("lycoris_") for k in net_sd.keys())
    net_has_lora_unet = any(k.startswith("lora_unet_") for k in net_sd.keys())

    formatted_sd = {}
    for k, v in sd.items():
        if net_has_lora_unet:
            new_k = k.replace("lycoris_", "lora_unet_", 1) if k.startswith("lycoris_") else k
        elif net_has_lycoris:
            new_k = k.replace("lora_unet_", "lycoris_", 1) if k.startswith("lora_unet_") else k
        else:
            new_k = k
        formatted_sd[new_k] = v

    if hasattr(network, "load_state_dict"):
        return network.load_state_dict(formatted_sd, strict=False)
    elif hasattr(network, "load_weights"):
        return network.load_weights(str(checkpoint_path))
    else:
        raise AttributeError("Provided network object does not support load_state_dict or load_weights")


def load_training_state(checkpoint_path: str | Path, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> dict:
    """Loads optimizer, scheduler, epoch, and step from corresponding .state.pt file if present."""
    p = Path(checkpoint_path)
    state_file = p.with_suffix(".state.pt") if not p.name.endswith(".state.pt") else p
    if not state_file.exists():
        return {}

    state = torch.load(str(state_file), map_location="cpu")
    if optimizer is not None and "optimizer_state_dict" in state and state["optimizer_state_dict"] is not None:
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            print("✅ Restored optimizer state.")
        except Exception as e:
            print(f"⚠️ Could not restore optimizer state: {e}")

    if scheduler is not None and "scheduler_state_dict" in state and state["scheduler_state_dict"] is not None:
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
            print("✅ Restored scheduler state.")
        except Exception as e:
            print(f"⚠️ Could not restore scheduler state: {e}")

    return {
        "epoch": state.get("epoch", 0),
        "global_step": state.get("global_step", 0),
    }