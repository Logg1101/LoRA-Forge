import gradio as gr
import tkinter as tk
from tkinter import filedialog

from core.state_manager import TrainingStateManager
from core.config_handler import save_config_to_file, load_config_from_file
from engine.trainer import run_training_loop

# Initialize central process controller
state_manager = TrainingStateManager()
current_config = load_config_from_file()


# --- NATIVE FILE BROWSER FUNCTIONS ---
def open_file_browser(current_path):
    """Opens a native Windows dialog to select a single file (like a .safetensors model)."""
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Base Model",
        filetypes=[("Safetensors", "*.safetensors"), ("All Files", "*.*")],
    )
    root.destroy()
    return file_path if file_path else current_path


def open_folder_browser(current_path):
    """Opens a native Windows dialog to select a directory."""
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Directory")
    root.destroy()
    return folder_path if folder_path else current_path


# ------------------------------------


def launch_training(
    project_name, base_model_path, model_type, dataset_dir, output_dir,
    network_type, rank, alpha, learning_rate, text_encoder_lr,
    text_encoder_training, optimizer, lr_scheduler, batch_size, epochs,
    max_train_steps, mixed_precision, resolution, cache_latents,
    grad_accum_steps, gradient_checkpointing, gradient_clip_norm,
    quantization, backup_steps, save_precision,
):
    """Packages all UI inputs into an explicit dict, saves, and starts the background process."""

    # Explicit mapping — immune to dict ordering bugs
    config_dict = {
        "project_name": project_name,
        "base_model_path": base_model_path,
        "model_type": model_type,
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "network_type": network_type,
        "rank": int(rank),
        "alpha": float(alpha),
        "learning_rate": float(learning_rate),
        "text_encoder_lr": float(text_encoder_lr),
        "text_encoder_training": bool(text_encoder_training),
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "max_train_steps": int(max_train_steps),
        "mixed_precision": mixed_precision,
        "resolution": int(resolution),
        "cache_latents": bool(cache_latents),
        "grad_accum_steps": int(grad_accum_steps),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "gradient_clip_norm": float(gradient_clip_norm),
        "quantization": quantization,
        "backup_steps": int(backup_steps),
        "save_precision": save_precision,
    }

    saved_config = save_config_to_file(config_dict)
    state_manager.start_training(saved_config, run_training_loop)
    return f"🚀 Training '{saved_config['project_name']}' started! Check your terminal for live logs."


def launch_ui():
    """Builds and launches the full-featured Gradio interface."""

    with gr.Blocks(title="LoRA Forge") as app:
        gr.Markdown("# 🔨 LoRA Forge")
        gr.Markdown(
            "A custom, offline trainer for LoRA, DoRA, LoHA, and LyCORIS "
            "with live pause/resume, auto-backups, latent caching, and Flux support."
        )

        with gr.Row():
            # LEFT SIDE: SETTINGS PANEL
            with gr.Column(scale=3):

                # 🔴 SECTION 1: CRITICAL PATHS & MODEL ARCHITECTURE
                gr.HTML("""
                <div style='border-left: 5px solid #FF4D4D; padding-left: 10px; margin-bottom: 12px;'>
                    <h3 style='color: #FF4D4D; margin: 0;'>🔴 Core Paths & Model Architecture</h3>
                </div>
                """)

                project_name = gr.Textbox(
                    label="Project Name", value=current_config.get("project_name")
                )

                with gr.Row():
                    base_model_path = gr.Textbox(
                        label="Base Model Path",
                        value=current_config.get("base_model_path"),
                        placeholder="Path to .safetensors or diffusers folder",
                        scale=4,
                    )
                    browse_model_btn = gr.Button("📄 Browse File", scale=1)
                    model_type = gr.Dropdown(
                        choices=["SD1.5", "SDXL", "Flux.1", "Flux Schnell", "Flux Dev"],
                        value=current_config.get("model_type"),
                        label="Architecture",
                        scale=2,
                    )

                with gr.Row():
                    dataset_dir = gr.Textbox(
                        label="Dataset Directory",
                        value=current_config.get("dataset_dir"),
                        scale=4,
                    )
                    browse_dataset_btn = gr.Button("📂 Browse Folder", scale=1)

                with gr.Row():
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value=current_config.get("output_dir"),
                        scale=4,
                    )
                    browse_output_btn = gr.Button("📂 Browse Folder", scale=1)

                gr.HTML("<hr style='border: 0.5px solid #333; margin: 20px 0;'>")

                # 🟠 SECTION 2: NETWORK ARCHITECTURE
                gr.HTML("""
                <div style='border-left: 5px solid #FFA500; padding-left: 10px; margin-bottom: 12px;'>
                    <h3 style='color: #FFA500; margin: 0;'>🟠 Network Architecture & Adapters</h3>
                </div>
                """)

                with gr.Row():
                    network_type = gr.Dropdown(
                        choices=["lora", "dora", "loha", "locon", "lycoris"],
                        value=current_config.get("network_type"),
                        label="Network Type",
                    )
                    rank = gr.Number(
                        label="Rank (Dim)",
                        value=current_config.get("rank"),
                        precision=0,
                    )
                    alpha = gr.Number(
                        label="Alpha", value=current_config.get("alpha")
                    )

                gr.HTML("<hr style='border: 0.5px solid #333; margin: 20px 0;'>")

                # 🟡 SECTION 3: HYPERPARAMETERS
                gr.HTML("""
                <div style='border-left: 5px solid #FFD700; padding-left: 10px; margin-bottom: 12px;'>
                    <h3 style='color: #FFD700; margin: 0;'>🟡 Hyperparameters & Optimization</h3>
                </div>
                """)

                with gr.Row():
                    learning_rate = gr.Number(
                        label="UNet Learning Rate",
                        value=current_config.get("learning_rate"),
                    )
                    text_encoder_lr = gr.Number(
                        label="Text Encoder Learning Rate",
                        value=current_config.get("text_encoder_lr"),
                    )

                text_encoder_training = gr.Checkbox(
                    label="Train Text Encoder(s)",
                    value=current_config.get("text_encoder_training", False),
                    info="Enable to fine-tune text encoders alongside the LoRA. Requires cache_latents OFF.",
                )

                with gr.Row():
                    optimizer = gr.Dropdown(
                        choices=["AdamW", "AdamW8bit", "Prodigy"],
                        value=current_config.get("optimizer"),
                        label="Optimizer",
                    )
                    lr_scheduler = gr.Dropdown(
                        choices=["cosine", "constant", "linear"],
                        value=current_config.get("lr_scheduler"),
                        label="Scheduler",
                    )

                with gr.Row():
                    batch_size = gr.Number(
                        label="Batch Size",
                        value=current_config.get("batch_size"),
                        precision=0,
                    )
                    epochs = gr.Number(
                        label="Epochs",
                        value=current_config.get("epochs"),
                        precision=0,
                    )
                    max_train_steps = gr.Number(
                        label="Max Steps (0 = Use Epochs)",
                        value=current_config.get("max_train_steps"),
                        precision=0,
                    )

                gr.HTML("<hr style='border: 0.5px solid #333; margin: 20px 0;'>")

                # 🟢 SECTION 4: HARDWARE
                gr.HTML("""
                <div style='border-left: 5px solid #2ECC71; padding-left: 10px; margin-bottom: 12px;'>
                    <h3 style='color: #2ECC71; margin: 0;'>🟢 Hardware Performance & Memory</h3>
                </div>
                """)

                with gr.Row():
                    resolution = gr.Number(
                        label="Training Resolution",
                        value=current_config.get("resolution"),
                        precision=0,
                    )
                    mixed_precision = gr.Dropdown(
                        choices=["bf16", "fp16", "fp8", "no"],
                        value=current_config.get("mixed_precision"),
                        label="Precision",
                    )
                    grad_accum_steps = gr.Number(
                        label="Gradient Accumulation Steps",
                        value=current_config.get("grad_accum_steps"),
                        precision=0,
                    )

                with gr.Row():
                    cache_latents = gr.Checkbox(
                        label="Cache Latents",
                        value=current_config.get("cache_latents"),
                        info="Pre-encode all images once — drastically speeds up training and frees VRAM.",
                    )
                    gradient_checkpointing = gr.Checkbox(
                        label="Gradient Checkpointing",
                        value=current_config.get("gradient_checkpointing", True),
                        info="Saves ~60% activation VRAM at ~30% speed cost. Essential for low-VRAM.",
                    )

                with gr.Row():
                    gradient_clip_norm = gr.Number(
                        label="Gradient Clip Norm (0 = off)",
                        value=current_config.get("gradient_clip_norm", 1.0),
                    )
                    quantization = gr.Dropdown(
                        choices=["none", "nf4", "int8"],
                        value=current_config.get("quantization", "none"),
                        label="Base Model Quantization",
                        info="NF4: ~4x VRAM savings (12GB Flux training). Requires diffusers-format model dir.",
                    )

                gr.HTML("<hr style='border: 0.5px solid #333; margin: 20px 0;'>")

                # 🔵 SECTION 5: BACKUPS
                gr.HTML("""
                <div style='border-left: 5px solid #3498DB; padding-left: 10px; margin-bottom: 12px;'>
                    <h3 style='color: #3498DB; margin: 0;'>🔵 Backups & Checkpoints</h3>
                </div>
                """)

                with gr.Row():
                    backup_steps = gr.Number(
                        label="Backup Interval (steps)",
                        value=current_config.get("backup_steps"),
                        precision=0,
                    )
                    save_precision = gr.Dropdown(
                        choices=["fp16", "bf16", "float32"],
                        value=current_config.get("save_precision"),
                        label="Save Precision",
                    )

            # RIGHT SIDE: CONTROLS
            with gr.Column(scale=2):
                gr.HTML("""
                <div style='border-left: 5px solid #9B59B6; padding-left: 10px; margin-bottom: 12px;'>
                    <h3 style='color: #9B59B6; margin: 0;'>🎮 Process Controller</h3>
                </div>
                """)

                status_box = gr.Textbox(
                    label="Execution Status",
                    interactive=False,
                    value="Idle — Ready to train.",
                )

                start_btn = gr.Button("▶️ Start Training", variant="primary")

                with gr.Row():
                    pause_btn = gr.Button("⏸️ Pause Engine")
                    resume_btn = gr.Button("⏯️ Resume Engine")

                stop_btn = gr.Button("🛑 Stop / Abort Process", variant="stop")

                gr.HTML("<hr style='border: 0.5px solid #333; margin: 20px 0;'>")

                gr.Markdown("""
                ### ℹ️ Quick Tips
                - **Pause** lets the current step finish, then waits.
                - **Stop** saves a final checkpoint, then exits.
                - **Backups** are auto-saved every N steps to `checkpoints/backups/`.
                - **Cache Latents** is the #1 speed boost — turn it ON.
                - **Prodigy** optimizer is LR-free (ignores learning rate).

                ### 🔧 12GB VRAM Flux Recipe
                Set: **NF4** quantization, **Cache Latents** ON,
                **Gradient Checkpointing** ON, **AdamW8bit** optimizer,
                **bf16** precision, resolution ≤ **768**.
                """)

        # --- EVENT LISTENERS FOR BROWSER BUTTONS ---
        browse_model_btn.click(
            fn=open_file_browser, inputs=[base_model_path], outputs=[base_model_path]
        )
        browse_dataset_btn.click(
            fn=open_folder_browser, inputs=[dataset_dir], outputs=[dataset_dir]
        )
        browse_output_btn.click(
            fn=open_folder_browser, inputs=[output_dir], outputs=[output_dir]
        )

        # Bundle all inputs — explicit list matching launch_training's signature
        all_inputs = [
            project_name, base_model_path, model_type, dataset_dir, output_dir,
            network_type, rank, alpha, learning_rate, text_encoder_lr,
            text_encoder_training, optimizer, lr_scheduler, batch_size, epochs,
            max_train_steps, mixed_precision, resolution, cache_latents,
            grad_accum_steps, gradient_checkpointing, gradient_clip_norm,
            quantization, backup_steps, save_precision,
        ]

        # --- PROCESS CONTROLLER EVENTS ---
        start_btn.click(fn=launch_training, inputs=all_inputs, outputs=[status_box])
        pause_btn.click(
            fn=lambda: [state_manager.pause_training(), "⏸️ Pausing..."][1],
            outputs=[status_box],
        )
        resume_btn.click(
            fn=lambda: [state_manager.resume_training(), "▶️ Resuming..."][1],
            outputs=[status_box],
        )
        stop_btn.click(
            fn=lambda: [state_manager.stop_training(), "🛑 Stopping..."][1],
            outputs=[status_box],
        )

    print("Launching LoRA Forge on http://127.0.0.1:7860")
    app.launch()