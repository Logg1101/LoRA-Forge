import gc
import os
import torch


def load_base_model(model_path, model_type="SDXL", mixed_precision="bf16", quantization="none"):
    """
    Loads model components offline, supporting single .safetensors files or local directories.
    Optionally quantizes the UNet/Transformer for low-VRAM training (NF4 / INT8).
    Deletes the pipeline after extracting components to free VRAM.
    """
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline, DDPMScheduler

    # 1. Map string to exact tensor precisions
    # fp8 is only supported on NVIDIA Ada/Hopper (sm_89+) — not on AMD ROCm.
    # Detect support at runtime and fall back gracefully to bf16.
    if mixed_precision == "fp8":
        if hasattr(torch, "float8_e4m3fn"):
            dtype = torch.float8_e4m3fn
        else:
            print("⚠️ fp8 not supported on this device/PyTorch build — falling back to bf16")
            dtype = torch.bfloat16
    else:
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "no": torch.float32,
        }
        dtype = dtype_map.get(mixed_precision, torch.float32)

    print(f"📦 Loading {model_type} base model from '{model_path}' in {dtype}...")

    # 2. Offline Path Detection
    is_single_file = os.path.isfile(model_path) and model_path.endswith(".safetensors")
    is_local_dir = os.path.isdir(model_path)
    local_kwargs = {"local_files_only": True} if is_local_dir else {}

    # 3. Quantization config (NF4 = ~4x VRAM savings, INT8 = ~2x)
    quant_config = _build_quant_config(quantization)

    # 4. Model Architecture Loaders
    if model_type == "SD1.5":
        pipe = _load_pipeline(
            StableDiffusionPipeline, model_path, is_single_file, dtype,
            local_kwargs, quant_config=None,  # SD1.5 rarely needs quantization
        )
        vae = pipe.vae
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        unet = pipe.unet
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        del pipe; gc.collect()
        return tokenizer, text_encoder, vae, unet, noise_scheduler

    elif model_type == "SDXL":
        pipe = _load_pipeline(
            StableDiffusionXLPipeline, model_path, is_single_file, dtype,
            local_kwargs, quant_config=quant_config,
        )
        vae = pipe.vae
        unet = pipe.unet
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        tokenizer, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
        text_encoder, text_encoder_2 = pipe.text_encoder, pipe.text_encoder_2

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)

        del pipe; gc.collect()
        return (tokenizer, tokenizer_2), (text_encoder, text_encoder_2), vae, unet, noise_scheduler

    elif "Flux" in model_type:
        pipe = _load_pipeline(
            FluxPipeline, model_path, is_single_file, dtype,
            local_kwargs, quant_config=quant_config,
        )
        vae = pipe.vae
        noise_scheduler = pipe.scheduler
        # Flux convention: tokenizer = CLIP, tokenizer_2 = T5
        tokenizer, tokenizer_2 = pipe.tokenizer, pipe.tokenizer_2
        text_encoder = pipe.text_encoder       # CLIP
        text_encoder_2 = pipe.text_encoder_2   # T5
        transformer = pipe.transformer

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)

        del pipe; gc.collect()
        return (tokenizer, tokenizer_2), (text_encoder, text_encoder_2), vae, transformer, noise_scheduler

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _build_quant_config(quantization):
    """
    Build a BitsAndBytesConfig for the requested quantization level.
    NF4/INT8 quantization via bitsandbytes is NVIDIA CUDA only.
    On AMD ROCm this is a no-op with a warning.
    """
    if quantization == "none":
        return None

    # bitsandbytes requires NVIDIA CUDA — it does not support AMD ROCm
    if not torch.cuda.is_available():
        print(f"⚠️ Quantization ({quantization}) requires NVIDIA CUDA — not available on this device.")
        print("   AMD ROCm users: you have enough VRAM (24GB) to train without quantization.")
        return None

    # Sanity-check that this is actually NVIDIA, not a ROCm build of torch
    # (some ROCm PyTorch builds report cuda as available via HIP shim)
    try:
        device_name = torch.cuda.get_device_name(0).lower()
        if "amd" in device_name or "radeon" in device_name:
            print(f"⚠️ Detected AMD GPU ({torch.cuda.get_device_name(0)}). NF4/INT8 quantization (bitsandbytes) is NVIDIA-only.")
            print("   Skipping quantization — your 24GB VRAM is sufficient without it.")
            return None
    except Exception:
        pass

    try:
        from diffusers import BitsAndBytesConfig
    except ImportError:
        print("⚠️ BitsAndBytesConfig not available in this diffusers version. Skipping quantization.")
        return None

    if quantization == "nf4":
        print("🔧 Enabling NF4 (4-bit) quantization — ~4x VRAM savings on the base model")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "int8":
        print("🔧 Enabling INT8 (8-bit) quantization — ~2x VRAM savings on the base model")
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        print(f"⚠️ Unknown quantization '{quantization}', skipping.")
        return None


def _load_pipeline(pipeline_cls, model_path, is_single_file, dtype, local_kwargs, quant_config=None):
    """
    Load a diffusers pipeline, handling single-file vs directory loading and
    optional quantization.
    """
    if is_single_file:
        # from_single_file doesn't support quantization_config in all diffusers versions.
        # Load in full precision first; for Flux on low-VRAM, recommend using directory format.
        if quant_config is not None:
            print("⚠️ Quantization with single-file loading may not be supported.")
            print("   For best results with NF4/INT8, use a diffusers-format model directory.")
            try:
                pipe = pipeline_cls.from_single_file(
                    model_path, torch_dtype=dtype, quantization_config=quant_config,
                )
                return pipe
            except TypeError:
                print("   Falling back to non-quantized single-file load.")

        pipe = pipeline_cls.from_single_file(model_path, torch_dtype=dtype)
    else:
        load_kwargs = {"torch_dtype": dtype, **local_kwargs}
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        pipe = pipeline_cls.from_pretrained(model_path, **load_kwargs)

    return pipe