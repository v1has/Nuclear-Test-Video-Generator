"""
scripts/04_preencode_latents.py

Stage 3 & 4 Latent Pre-Encoding Engine.
Pre-encodes standardized 16-frame explosion clips into Stable Diffusion 1.5
latent tensors using AutoencoderKL.

Pre-caching latents saves GPU memory and eliminates redundant per-epoch VAE forward
passes during UNet / Motion LoRA training on Kaggle.

Input:
- data/manifest.csv (with paths to processed 16-frame MP4 clips)

Output:
- latents/<clip_id>.pt: Saved PyTorch tensor of shape (4, 16, H/8, W/8)
- Updated data/manifest.csv with latent_path column

Usage:
  python scripts/04_preencode_latents.py --limit 10
  python scripts/04_preencode_latents.py --device cpu
  python scripts/04_preencode_latents.py --all
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from diffusers import AutoencoderKL

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LATENTS_DIR = os.path.join(PROJECT_ROOT, "latents")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")

# Standard SD 1.5 VAE scaling factor
VAE_SCALING_FACTOR = 0.18215


def load_vae(model_name="runwayml/stable-diffusion-v1-5", device="cpu", torch_dtype=torch.float32):
    """Load pre-trained Stable Diffusion 1.5 VAE."""
    print(f"[1/3] Loading SD 1.5 AutoencoderKL from {model_name}...")
    try:
        vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=torch_dtype
        )
    except Exception as e:
        print(f"Warning: Failed loading from {model_name}/vae ({e}). Attempting stabilityai/sd-vae-ft-mse...")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch_dtype
        )
    vae.to(device)
    vae.eval()
    print(f"      VAE loaded successfully on device: {device} ({torch_dtype})")
    return vae


def read_clip_frames(clip_path):
    """
    Read 16 frames from MP4 clip, convert BGR to RGB, normalize to [-1, 1],
    and return tensor of shape (16, 3, H, W).
    """
    full_path = os.path.join(PROJECT_ROOT, clip_path) if not os.path.isabs(clip_path) else clip_path
    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        return None
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize to [-1, 1]
        norm = (rgb.astype(np.float32) / 127.5) - 1.0
        # Transpose (H, W, C) -> (C, H, W)
        chw = np.transpose(norm, (2, 0, 1))
        frames.append(chw)
        
    cap.release()
    if not frames:
        return None
        
    # Stack along time dimension: (T, C, H, W)
    tensor = torch.tensor(np.array(frames), dtype=torch.float32)
    return tensor


def preencode_latents(vae, manifest_df, device="cpu", torch_dtype=torch.float32, limit=None):
    """
    Encode clips into latent tensors and save as .pt files.
    """
    os.makedirs(LATENTS_DIR, exist_ok=True)
    
    targets = manifest_df.copy()
    if limit:
        targets = targets.head(limit)
        
    print(f"[2/3] Pre-encoding {len(targets)} video clips into latent tensors...")
    
    encoded_count = 0
    start_time = time.time()
    
    for idx, row in tqdm(targets.iterrows(), total=len(targets), desc="Encoding Latents"):
        clip_id = row["clip_id"]
        clip_path = row["clip_path"]
        
        latent_filename = f"{clip_id}.pt"
        latent_path = os.path.join(LATENTS_DIR, latent_filename)
        rel_latent_path = os.path.relpath(latent_path, PROJECT_ROOT)
        
        # Check if already encoded
        if os.path.exists(latent_path):
            manifest_df.loc[manifest_df["clip_id"] == clip_id, "latent_path"] = rel_latent_path
            encoded_count += 1
            continue
            
        frames_tensor = read_clip_frames(clip_path)
        if frames_tensor is None or frames_tensor.shape[0] < 16:
            print(f"Warning: Skipping {clip_id} (could not read 16 frames).")
            continue
            
        # Ensure 16 frames
        frames_tensor = frames_tensor[:16]
        # Shape: (16, 3, H, W)
        frames_tensor = frames_tensor.to(device=device, dtype=torch_dtype)
        
        with torch.no_grad():
            # Encode frames in micro-batches (e.g. 2 frames at a time) to prevent OOM
            # On CPU, 16 full 512x512 frames simultaneously can exceed memory allocator limits
            chunk_size = 2 if device.startswith("cpu") else 4
            latent_chunks = []
            for chunk in frames_tensor.split(chunk_size):
                chunk_latent = vae.encode(chunk).latent_dist.sample()
                latent_chunks.append(chunk_latent)
            latents = torch.cat(latent_chunks, dim=0) * VAE_SCALING_FACTOR
            
        # Permute (16, 4, H/8, W/8) -> (4, 16, H/8, W/8) for AnimateDiff temporal attention
        # AnimateDiff standard motion module shape: (Channels, Frames, Height, Width)
        latents_video = latents.permute(1, 0, 2, 3).cpu()
        
        # Save PyTorch tensor
        torch.save(latents_video, latent_path)
        manifest_df.loc[manifest_df["clip_id"] == clip_id, "latent_path"] = rel_latent_path
        encoded_count += 1
        
    total_time = time.time() - start_time
    fps = (encoded_count * 16) / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 65)
    print("       LATENT PRE-ENCODING BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Clips encoded: {encoded_count}/{len(targets)}")
    print(f"Total time elapsed: {total_time:.2f} seconds")
    print(f"Throughput: {fps:.2f} frames/sec ({total_time/max(1, encoded_count)*1000:.1f} ms/clip)")
    if torch.cuda.is_available() and device.startswith("cuda"):
        print(f"Peak VRAM used: {torch.cuda.max_memory_allocated() / (1024*1024):.1f} MB")
        
    return manifest_df


def main():
    parser = argparse.ArgumentParser(description="Stage 3 & 4 Latent Pre-Encoding Engine")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model ID for VAE")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips to encode for dry run")
    parser.add_argument("--all", action="store_true", help="Encode all clips in manifest.csv")
    
    args = parser.parse_args()
    
    if not os.path.exists(MANIFEST_CSV):
        print(f"Error: {MANIFEST_CSV} not found! Run scripts/03_cut_and_filter.py first.")
        sys.exit(1)
        
    manifest_df = pd.read_csv(MANIFEST_CSV)
    print(f"Loaded {len(manifest_df)} clips from {MANIFEST_CSV}")
    
    # Auto-detect device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32
    
    vae = load_vae(model_name=args.model, device=device, torch_dtype=torch_dtype)
    
    limit = args.limit if args.limit else (None if args.all else 10)
    updated_df = preencode_latents(vae, manifest_df, device=device, torch_dtype=torch_dtype, limit=limit)
    
    updated_df.to_csv(MANIFEST_CSV, index=False)
    print(f"[3/3] Updated manifest with latent paths saved to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
