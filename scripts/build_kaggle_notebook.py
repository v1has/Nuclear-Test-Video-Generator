"""
scripts/build_kaggle_notebook.py

Generates training/kaggle_train_pipeline.ipynb for Kaggle execution (T4/P100 GPU).
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(PROJECT_ROOT, "training", "kaggle_train_pipeline.ipynb")

def make_cell(cell_type, source, execution_count=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")] if isinstance(source, str) else source
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = []
    return cell

def build_notebook():
    cells = []
    
    # Cell 1: Markdown Title & Abstract
    cells.append(make_cell("markdown", """# Parameter-Controlled Archival Nuclear Test Video Generator
### Two-Phase LoRA Training Pipeline (Kaggle T4 / P100 GPU)

**Pipeline Architecture:**
1. **Phase 1 (Spatial LoRA):** Fine-tunes SD 1.5 UNet spatial self/cross-attention projections (`to_k`, `to_v`, `to_q`, `to_out.0`) on static frames to ground visual appearance:
   - Burst geometry and cloud morphology
   - Nevada Test Site desert vs. Pacific Proving Grounds ocean
   - Historical 16mm film grain, exposure, and color degradation
2. **Phase 2 (Motion LoRA):** Freezes spatial UNet (with Spatial LoRA active), fine-tunes AnimateDiff temporal *self-attention* layers on 16-frame latent sequences (`4, 16, 64, 64`) to learn fluid explosion dynamics:
   - Shockwave propagation & Wilson cloud flash
   - Toroidal vortex expansion & stem suction
   - Base surge progression in underwater/shallow bursts
3. **Stage 6 Evaluation Suite:** Multi-task 3D classifier measuring control accuracy against majority-class baselines.
"""))

    # Cell 2: Environment Setup
    cells.append(make_cell("code", """# Step 1: Environment Setup & Package Installation
!pip install -q diffusers transformers accelerate peft pyyaml scenedetect opencv-python imageio imageio-ffmpeg tqdm
!nvidia-smi
"""))

    # Cell 3: GPU Verification
    cells.append(make_cell("code", """# Step 2: GPU and PyTorch Verification
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
"""))

    # Cell 4: Project Directory Structure & Git/Files setup
    cells.append(make_cell("code", """# Step 3: Project Setup in Kaggle Working Directory
import os
import shutil

WORKDIR = "/kaggle/working/Nuclear-Test-Video-Generator"
if not os.path.exists(WORKDIR):
    # Clone or link project code
    !git clone https://github.com/v1has/Nuclear-Test-Video-Generator.git {WORKDIR} || mkdir -p {WORKDIR}

os.chdir(WORKDIR)
print("Current Working Directory:", os.getcwd())
!ls -lh
"""))

    # Cell 5: Verify Manifest & Pre-Encode Latents
    cells.append(make_cell("code", """# Step 4: Pre-encode Latents on GPU (if not already cached)
# GPU pre-encoding takes ~100-200 ms per clip, saving gigabytes of per-epoch compute
!python scripts/04_preencode_latents.py --all --device cuda
"""))

    # Cell 6: Phase 1 Spatial LoRA Training
    cells.append(make_cell("code", """# Step 5: Phase 1 (Spatial LoRA) Training
# Trains spatial cross-attention layers on static frames
!python training/01_train_spatial_lora.py \\
    --config training/config.yaml \\
    --max_train_steps 2000 \\
    --device cuda
"""))

    # Cell 7: Phase 2 Motion LoRA Training
    cells.append(make_cell("code", """# Step 6: Phase 2 (Motion LoRA) Training
# Freezes spatial UNet + Spatial LoRA, trains temporal self-attention in AnimateDiff motion module
!python training/02_train_motion_lora.py \\
    --config training/config.yaml \\
    --max_train_steps 3000 \\
    --device cuda
"""))

    # Cell 8: Inference & Sample Video Generation
    cells.append(make_cell("code", """# Step 7: Sample Video Generation Across Burst Types
!python inference/generate.py \\
    --burst_type underwater \\
    --yield_tier tactical-yield \\
    --setting tropical-ocean \\
    --phase expansion \\
    --film_look colour-restored \\
    --output samples/underwater_sample.mp4 \\
    --device cuda

!python inference/generate.py \\
    --burst_type surface-tower \\
    --yield_tier intermediate-yield \\
    --setting desert-flats \\
    --phase expansion \\
    --film_look high-contrast-monochrome \\
    --output samples/surface_tower_sample.mp4 \\
    --device cuda
"""))

    # Cell 9: Stage 6 Quantitative Control Evaluation
    cells.append(make_cell("code", """# Step 8: Stage 6 Quantitative Control Evaluation Suite
# Evaluates generated samples against trained 3D classifier to verify >25% margin over majority baseline
!python scripts/05_train_classifier.py --train --epochs 5 --device cuda
!python scripts/05_train_classifier.py --evaluate_synthetic --synthetic_dir samples/ --device cuda
"""))

    # Cell 10: Packaging Output Weights
    cells.append(make_cell("code", """# Step 9: Package Checkpoints for Download
!zip -r /kaggle/working/nuclear_generator_lora_weights.zip \\
    checkpoints/phase1_spatial/final_spatial_lora \\
    checkpoints/phase2_motion/final_motion_lora \\
    models/control_classifier.pt \\
    data/synthetic_control_report.json

print("Package created at /kaggle/working/nuclear_generator_lora_weights.zip")
"""))

    notebook_json = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(notebook_json, f, indent=2)
    print(f"Generated Kaggle notebook at: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    build_notebook()
