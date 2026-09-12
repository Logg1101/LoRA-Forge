# LoRA Forge

<div align="center">

**High-Performance LoRA & LyCORIS Training Workstation for Diffusion Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 33 Passed](https://img.shields.io/badge/Tests-33%20Passed-brightgreen.svg)](#running-tests)

</div>

---

## Overview

**LoRA Forge** is a streamlined, professional-grade desktop training workstation for fine-tuning diffusion models using Low-Rank Adaptation (LoRA) and LyCORIS. Built with **PySide6** and **Accelerate**, LoRA Forge combines mathematical precision with a clean, modern UI featuring live training pause/resume, aspect-ratio bucketing, UNet block weighting, Prodigy adaptive optimization, and embedded TensorBoard visualization.

---

## Key Features

- 🛑 **Live Engine Control**: Pause, resume, or cleanly terminate training at any step without corrupting weights or optimizer states.
- 🎯 **Diffusion Math**: Discrete forward DDPM noise scheduling and velocity formulation ported from verified implementations. Guarantees clean, noise-free image generation across all diffusion samplers (Euler, Euler A, DPM++ 2M Karras, Heun, DDIM, etc.) at standard inference strengths (0.7 - 1.0).
- 🧬 **Multi-Architecture Support**:
  - **Stable Diffusion 1.5**
  - **Stable Diffusion XL (SDXL)** with dual text encoders (CLIP ViT-L & OpenCLIP ViT-bigG) and pooled text projections
  - **Flux** and **Pony** models
- ⚡ **LyCORIS Adapter Suite**:
  - Standard **LoRA** (Low-Rank Adaptation)
  - **DoRA** (Weight-Decomposed Low-Rank Adaptation)
  - **LoHa** (Hadamard Product)
  - **LoCon** (Convolutional LoRA for style/composition transfer)
- 🎚️ **3-Tier UNet Block Multipliers**:
  - Independent multipliers for **Down Blocks** (Input), **Mid Block**, and **Up Blocks** (Output)
  - Zero-weight layer freezing (`requires_grad = False`) to prevent character identity bleeding into base styles
- 🏷️ **Advanced Caption & Tag Dynamics**:
  - Tag-level comma shuffling per epoch
  - Prefix **keep-tokens** protection (protect trigger words)
  - Individual **tag dropout** rate and full caption dropout
  - Dynamic caption processing with RAM latent caching
- 🚀 **Optimizers & Loss Formulations**:
  - **Prodigy** adaptive step-size with D-estimate coefficients, bias correction, and relative parameter-group LR scaling
  - **AdamW** and **AdamW 8-bit** (via `bitsandbytes`)
  - **Min-SNR Gamma** weighting (with v-prediction and epsilon support)
  - **Huber** robust loss metric
- 📊 **Embedded TensorBoard & Telemetry**:
  - Integrated TensorBoard dashboard inside the application via `QWebEngineView`
  - Automatic port allocation, auto-start on training launch, and one-click external browser viewing
  - Live loss graph widget, ETA calculator, and GPU VRAM telemetry

---

## Workspace Layout

LoRA Forge organizes complex training options into 5 unified, scrollable tabs:

1. **📁 Model & Data**: Base model path, model architecture, output directories, aspect-ratio bucketing rules, repeats, and dataset integrity validator.
2. **🧬 Architecture**: Adapter type (LoRA/DoRA/LoHa/LoCon), rank, alpha, conv rank/alpha, rank dropout, and UNet block LR multipliers.
3. **⚡ Training & HW**: Epochs, batch size, gradient accumulation, learning rates, Prodigy/AdamW optimizer, loss objective (MSE/Huber/Min-SNR), precision (fp16/bf16), latent caching, and base model quantization (NF4).
4. **📊 TensorBoard**: Interactive real-time metrics dashboard embedded in the application.
5. **💾 Output & Logs**: Validation preview generation prompts, checkpoint backup/snapshot pruning, preset configuration saving, and system console.

---

## Installation

### Prerequisites

- **Operating System**: Windows 10/11 or Linux
- **Python**: 3.10, 3.11, or 3.12
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended for SD1.5, 12GB+ for SDXL with NF4/caching)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Logg1101/LoRA-Forge.git
   cd LoRA-Forge
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Quickstart

### Launching the Application

- **Windows**: Double-click `run.bat` or run:
  ```bash
  python main.py
  ```
- **Linux**:
  ```bash
  python3 main.py
  ```

### Training Workflow

1. In **Model & Data**, select your base model checkpoint (`.safetensors` or Diffusers directory) and your image dataset folder.
2. In **Architecture**, select your adapter type (default: `lora`, rank `16`, alpha `8.0`).
3. In **Training & HW**, set your target epochs, learning rate, and optimizer (e.g. `AdamW8bit` or `Prodigy`).
4. Click **Start Training** in the top bar.
5. Monitor live loss convergence and hardware metrics in the **TensorBoard** tab.

---

## Running Tests

LoRA Forge includes a comprehensive test suite covering diffusion math, DDPM step verification, loss functions, optimizer setups, and UI synchronization:

```bash
pytest -v
```

All 33 test cases run in seconds and require no GPU.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
