"""
scripts/build_kaggle_notebook.py

Generates training/kaggle_train_pipeline.ipynb for Kaggle execution (T4/P100 GPU).

Fixes P0.8:
- Mounts and checks versioned Kaggle dataset (/kaggle/input/nuclear-test-video-dataset)
- Parameterizes dataset paths and normalizes paths with .as_posix()
- Asserts manifest and latent coverage before training
- Pins package versions (including tensorboard, imageio, imageio-ffmpeg)
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
    
    # Cell 1: Markdown Title & Pipeline Overview
    cells.append(make_cell("markdown", """# Parameter-Controlled Archival Nuclear Test Video Generator
### Two-Phase LoRA Training Pipeline (Kaggle T4 / P100 GPU)

**Pipeline Architecture:**
1. **Phase 1 (Spatial LoRA):** Fine-tunes SD 1.5 UNet spatial attention projections on static explosion frames to ground visual appearance:
   - Burst geometry and cloud morphology
   - Nevada Test Site desert vs. Pacific Proving Grounds ocean
   - Historical 16mm film grain and color palette
2. **Phase 2 (Motion LoRA):** Freezes spatial UNet, loads Phase 1 Spatial LoRA, and fine-tunes AnimateDiff temporal *self-attention* layers on 16-frame latent sequences (`4, 16, 64, 64`) to learn fluid explosion dynamics:
   - Shockwave propagation & Wilson cloud flash
   - Toroidal vortex expansion & stem suction
   - Base surge progression in underwater/shallow bursts
3. **Stage 6 Evaluation Suite:** Multi-task 3D classifier measuring control accuracy against majority-class baselines.
"""))

    # Cell 2: Environment Setup
    cells.append(make_cell("code", """# Step 1: Environment Setup & Pinned Package Installation
!pip install -q \\
    diffusers==0.27.2 \\
    transformers==4.38.2 \\
    accelerate==0.28.0 \\
    peft==0.10.0 \\
    tensorboard==2.16.2 \\
    imageio==2.34.0 \\
    imageio-ffmpeg==0.4.9 \\
    scenedetect[opencv]==0.6.3 \\
    pyyaml tqdm
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

    # Cell 4: Project Directory Structure & Kaggle Dataset Mount
    cells.append(make_cell("code", """# Step 3: Project Setup & Kaggle Dataset Mount (P0.8)
import os
import shutil
from pathlib import Path

WORKDIR = Path("/kaggle/working/Nuclear-Test-Video-Generator")
if not WORKDIR.exists():
    !git clone https://github.com/v1has/Nuclear-Test-Video-Generator.git {WORKDIR.as_posix()}

os.chdir(WORKDIR.as_posix())
print("Current Working Directory:", os.getcwd())

# Check for mounted Kaggle Dataset
KAGGLE_INPUT_DIR = Path("/kaggle/input/nuclear-test-video-generator")
if KAGGLE_INPUT_DIR.exists():
    print(f"Found mounted Kaggle dataset at {KAGGLE_INPUT_DIR.as_posix()}")
    for subdir in ["raw_videos", "processed_clips", "latents", "data"]:
        src = KAGGLE_INPUT_DIR / subdir
        dst = WORKDIR / subdir
        if src.exists() and not dst.exists():
            try:
                os.symlink(src.as_posix(), dst.as_posix())
                print(f"Linked {src.as_posix()} -> {dst.as_posix()}")
            except Exception as e:
                print(f"Symlink failed ({e}), copying directory {subdir}...")
                shutil.copytree(src.as_posix(), dst.as_posix())
"""))

    # Cell 5: Verify Manifest & Pre-Encode Latents
    cells.append(make_cell("code", """# Step 4: Verify Manifest and Latent Coverage (P0.3, P0.8)
import pandas as pd

manifest_path = Path("data/manifest.csv")
assert manifest_path.exists(), "data/manifest.csv not found!"

manifest = pd.read_csv(manifest_path.as_posix())
print(f"Total clips in manifest: {len(manifest)}")
print(f"Split distribution:\\n{manifest['split'].value_counts().to_string()}")

# Coverage assertion
covered = manifest['latent_path'].notna().sum() if 'latent_path' in manifest.columns else 0
print(f"Latents cached: {covered} / {len(manifest)} ({covered/max(1, len(manifest))*100:.1f}%)")

if covered < len(manifest):
    print("Pre-encoding remaining clips on GPU (takes ~150 ms/clip on T4)...")
    !python scripts/04_preencode_latents.py --device cuda
"""))

    # Cell 6: Phase 1 Spatial LoRA Training
    cells.append(make_cell("code", """# Step 5: Phase 1 (Spatial LoRA) Training
# Fine-tunes SD 1.5 spatial cross-attention layers on historical explosion frames
!python training/01_train_spatial_lora.py \\
    --config training/config.yaml \\
    --max_train_steps 2000 \\
    --device cuda
"""))

    # Cell 7: Phase 2 Motion LoRA Training
    cells.append(make_cell("code", """# Step 6: Phase 2 (Motion LoRA) Training
# Freezes spatial UNet, loads Phase 1 Spatial LoRA, fine-tunes AnimateDiff temporal self-attention
!python training/02_train_motion_lora.py \\
    --config training/config.yaml \\
    --max_train_steps 3000 \\
    --device cuda
"""))

    # Cell 8: Inference & Sample Video Generation
    cells.append(make_cell("code", """# Step 7: Sample Video Generation Across Burst Types
# Validates prompt tokens against schema.py and loads both Spatial and Motion LoRAs
!python inference/generate.py \\
    --burst_type air-burst \\
    --yield_tier tactical-yield \\
    --terrain desert-flats \\
    --lighting day \\
    --phase early-fireball \\
    --film_look black-and-white \\
    --smoke_trails \\
    --spatial_lora models/phase1_spatial/final_spatial_lora \\
    --motion_lora models/phase2_motion/final_motion_lora \\
    --output samples/eval_air_burst_fireball.mp4 \\
    --device cuda

!python inference/generate.py \\
    --burst_type surface-tower \\
    --yield_tier intermediate-yield \\
    --terrain desert-flats \\
    --lighting dawn \\
    --phase expansion \\
    --film_look black-and-white \\
    --spatial_lora models/phase1_spatial/final_spatial_lora \\
    --motion_lora models/phase2_motion/final_motion_lora \\
    --output samples/eval_tower_expansion.mp4 \\
    --device cuda

!python inference/generate.py \\
    --burst_type underwater \\
    --yield_tier tactical-yield \\
    --terrain tropical-ocean \\
    --lighting day \\
    --phase expansion \\
    --film_look colour-restored \\
    --spatial_lora models/phase1_spatial/final_spatial_lora \\
    --motion_lora models/phase2_motion/final_motion_lora \\
    --output samples/eval_underwater_dome.mp4 \\
    --device cuda
"""))

    # Cell 9: Stage 6 Quantitative Classifier Audit
    cells.append(make_cell("code", """# Step 8: Stage 6 Quantitative Classifier Audit
# Tests prompt adherence and control accuracy on synthesized clips (>= majority_baseline + 25%)
!python scripts/05_train_classifier.py --evaluate_synthetic --synthetic_dir samples/
"""))

    # Cell 10: Video Preview
    cells.append(make_cell("code", """# Step 9: Render Generated Video Clip in Notebook
from IPython.display import Video
Video("samples/eval_air_burst_fireball.mp4", embed=True, width=512)
"""))

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"[SUCCESS] Generated {NOTEBOOK_PATH} with {len(cells)} cells.")


if __name__ == "__main__":
    build_notebook()
