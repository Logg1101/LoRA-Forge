"""
Automated sample image generator during training.
Loads base model components or reuses existing unet/transformer to render test prompts with fixed seed.
"""
import os
from pathlib import Path
import torch
from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
    UniPCMultistepScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
)

SCHEDULER_REGISTRY = {
    "euler": lambda cfg: EulerDiscreteScheduler.from_config(cfg),
    "euler_a": lambda cfg: EulerAncestralDiscreteScheduler.from_config(cfg),
    "dpm++ 2m karras": lambda cfg: DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True),
    "dpm++ 2m sde karras": lambda cfg: DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
    "ddim": lambda cfg: DDIMScheduler.from_config(cfg),
    "lms": lambda cfg: LMSDiscreteScheduler.from_config(cfg),
    "heun": lambda cfg: HeunDiscreteScheduler.from_config(cfg),
    "uni_pc": lambda cfg: UniPCMultistepScheduler.from_config(cfg),
    "kdpm2": lambda cfg: KDPM2DiscreteScheduler.from_config(cfg),
    "pndm": lambda cfg: PNDMScheduler.from_config(cfg),
}


def generate_samples(
    unet,
    vae,
    tokenizer,
    text_encoder,
    config: dict,
    current_step: int,
    current_epoch: int,
    tokenizer_2=None,
    text_encoder_2=None,
    device=None,
):
    """
    Renders sample images using current LoRA weights and saves them to disk.
    Gracefully catches OOMs to prevent interrupting the training run.
    """
    if not config.get("enable_samples", False):
        return

    prompts = config.get("sample_prompts", ["portrait, cinematic lighting"])
    if not prompts:
        return

    output_dir = Path(config.get("output_dir", "checkpoints/final")) / "samples"
    step_dir = output_dir / f"step_{current_step}_epoch_{current_epoch}"
    step_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎨 Rendering {len(prompts)} sample image(s) at step {current_step}...")

    model_type = config.get("model_type", "SDXL")
    is_sdxl = model_type == "SDXL"
    is_flux = "Flux" in model_type
    sample_res = config.get("sample_resolution", 512)
    sample_steps = config.get("sample_steps", 28)
    sample_cfg = config.get("sample_cfg", 7.0)
    seed = config.get("sample_seed", 42)
    neg_prompt = config.get("sample_negative_prompt", "")

    dev = device or (unet.device if hasattr(unet, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    was_training = unet.training
    vae_device = getattr(vae, "device", "cpu")
    te1_device = getattr(text_encoder, "device", "cpu")
    te2_device = getattr(text_encoder_2, "device", "cpu") if text_encoder_2 is not None else None

    try:
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
            FluxPipeline,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            DDIMScheduler,
            LMSDiscreteScheduler,
            HeunDiscreteScheduler,
            UniPCMultistepScheduler,
            PNDMScheduler,
            FlowMatchEulerDiscreteScheduler,
        )

        generator = torch.Generator(device="cpu").manual_seed(seed)
        unet.eval()
        vae.to(dev)
        text_encoder.to(dev)
        if text_encoder_2 is not None:
            text_encoder_2.to(dev)

        # Build inference scheduler for sample preview without affecting training noise scheduler
        sampler_choice = str(config.get("sample_sampler", "euler")).lower().strip()
        default_scheduler_config = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "num_train_timesteps": 1000,
            "steps_offset": 1,
        }
        if config.get("v_prediction", False):
            default_scheduler_config["prediction_type"] = "v_prediction"

        factory = SCHEDULER_REGISTRY.get(sampler_choice)
        if factory is None:
            print(f"⚠️ Unknown sampler '{sampler_choice}' — falling back to Euler")
            factory = SCHEDULER_REGISTRY["euler"]
        inf_scheduler = factory(default_scheduler_config)

        with torch.inference_mode():
            if is_sdxl:
                pipe = StableDiffusionXLPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    unet=unet,
                    scheduler=inf_scheduler,
                )
                pipe.set_progress_bar_config(disable=True)
                pipe.to(dev)

                for idx, p_text in enumerate(prompts):
                    image = pipe(
                        prompt=p_text,
                        negative_prompt=neg_prompt,
                        num_inference_steps=sample_steps,
                        guidance_scale=sample_cfg,
                        width=sample_res,
                        height=sample_res,
                        generator=generator,
                    ).images[0]

                    save_path = step_dir / f"sample_{idx + 1}.png"
                    image.save(save_path)
                    print(f"   Saved: {save_path.name}")

                del pipe

            elif is_flux:
                flux_scheduler = FlowMatchEulerDiscreteScheduler.from_config({
                    "num_train_timesteps": 1000,
                    "shift": 3.0,
                })
                pipe = FluxPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    text_encoder_2=text_encoder_2,
                    tokenizer=tokenizer,
                    tokenizer_2=tokenizer_2,
                    transformer=unet,
                    scheduler=flux_scheduler,
                )
                pipe.set_progress_bar_config(disable=True)
                pipe.to(dev)

                for idx, p_text in enumerate(prompts):
                    image = pipe(
                        prompt=p_text,
                        num_inference_steps=sample_steps,
                        guidance_scale=sample_cfg,
                        width=sample_res,
                        height=sample_res,
                        generator=generator,
                    ).images[0]

                    save_path = step_dir / f"sample_{idx + 1}.png"
                    image.save(save_path)
                    print(f"   Saved: {save_path.name}")

                del pipe

            else:
                # SD1.5 and general 2D diffusion models
                pipe = StableDiffusionPipeline(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    scheduler=inf_scheduler,
                )
                pipe.set_progress_bar_config(disable=True)
                pipe.to(dev)

                for idx, p_text in enumerate(prompts):
                    image = pipe(
                        prompt=p_text,
                        negative_prompt=neg_prompt,
                        num_inference_steps=sample_steps,
                        guidance_scale=sample_cfg,
                        width=sample_res,
                        height=sample_res,
                        generator=generator,
                    ).images[0]

                    save_path = step_dir / f"sample_{idx + 1}.png"
                    image.save(save_path)
                    print(f"   Saved: {save_path.name}")

                del pipe

    except Exception as e:
        print(f"⚠️ Sample generation skipped due to error: {e}")
    finally:
        unet.train(was_training)
        # Restore offloaded device placement if VAE/TE were on CPU
        if str(vae_device) == "cpu":
            vae.to("cpu")
        if str(te1_device) == "cpu":
            text_encoder.to("cpu")
        if te2_device is not None and str(te2_device) == "cpu":
            text_encoder_2.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

