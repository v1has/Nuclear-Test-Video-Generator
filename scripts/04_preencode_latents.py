"""
scripts/04_preencode_latents.py

Stage 3 & 4 Latent Pre-Encoding Engine.
Encodes standardized 16-frame training video clips into compressed SD 1.5 latent space
using the AutoencoderKL model (runwayml/stable-diffusion-v1-5).

Fixes P0.3, P1 items, and Non-Negotiable Rules:
- Defaults to encoding ALL clips in manifest (asserts 100% latent coverage)
- Uses latent_dist.mode() for deterministic, reproducible latent representations
- Strict POSIX '/' paths written to manifest.csv
- Prevents shape mixing and batch crashes
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import time
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from diffusers import AutoencoderKL

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LATENTS_DIR = os.path.join(PROJECT_ROOT, "latents")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")

VAE_SCALING_FACTOR = 0.18215


def load_vae(model_name="runwayml/stable-diffusion-v1-5", device="cpu", torch_dtype=torch.float32):
    """Load pre-trained AutoencoderKL."""
    print(f"[1/3] Loading AutoencoderKL from {model_name}...")
    vae = AutoencoderKL.from_pretrained(
        model_name,
        subfolder="vae",
        torch_dtype=torch_dtype
    )
    vae.to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print(f"VAE successfully loaded onto {device} ({torch_dtype}).")
    return vae


def read_clip_frames(clip_path):
    """Read all frames from an MP4 clip, normalize to [-1, 1], return (16, 3, H, W)."""
    posix_path = str(clip_path).replace("\\", "/")
    full_path = os.path.join(PROJECT_ROOT, posix_path) if not os.path.isabs(posix_path) else posix_path
    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        return None
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 127.5) - 1.0
        chw = np.transpose(norm, (2, 0, 1))
        frames.append(chw)
        
    cap.release()
    if not frames:
        return None
        
    tensor = torch.tensor(np.array(frames), dtype=torch.float32)
    return tensor


def preencode_latents(vae, manifest_df, device="cpu", torch_dtype=torch.float32, limit=None, use_mode=True):
    """Encode clips into latent tensors and save as .pt files with 100% coverage."""
    os.makedirs(LATENTS_DIR, exist_ok=True)
    
    targets = manifest_df.copy()
    manifest_df["latent_path"] = manifest_df["latent_path"].astype(object)
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
        rel_latent_path = f"latents/{latent_filename}"  # Strict POSIX '/'
        
        # Check if already encoded
        if os.path.exists(latent_path):
            manifest_df.loc[manifest_df["clip_id"] == clip_id, "latent_path"] = rel_latent_path
            encoded_count += 1
            continue
            
        frames_tensor = read_clip_frames(clip_path)
        if frames_tensor is None or frames_tensor.shape[0] < 16:
            print(f"Warning: Skipping {clip_id} (could not read 16 frames).")
            continue
            
        frames_tensor = frames_tensor[:16].to(device=device, dtype=torch_dtype)
        
        with torch.no_grad():
            chunk_size = 2 if device.startswith("cpu") else 4
            latent_chunks = []
            for chunk in frames_tensor.split(chunk_size):
                dist = vae.encode(chunk).latent_dist
                # Use mode() for deterministic caching (P1 item)
                chunk_latent = dist.mode() if use_mode and hasattr(dist, "mode") else dist.sample()
                latent_chunks.append(chunk_latent)
            latents = torch.cat(latent_chunks, dim=0) * VAE_SCALING_FACTOR
            
        # Permute (16, 4, H/8, W/8) -> (4, 16, H/8, W/8) for AnimateDiff
        latents_video = latents.permute(1, 0, 2, 3).cpu()
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
    return manifest_df


def main():
    parser = argparse.ArgumentParser(description="Stage 3 & 4 Latent Pre-Encoding Engine")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model ID for VAE")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips to encode")
    parser.add_argument("--all", action="store_true", default=True, help="Encode all clips in manifest (default)")
    parser.add_argument("--assert-100", "--assert_100", dest="assert_100", action="store_true", default=True,
                        help="Fail if latent coverage is not 100%% (default: ON, AGENTS.md P0.3).")
    parser.add_argument("--allow-partial", dest="assert_100", action="store_false",
                        help="Allow partial latent coverage (cache-building only; never for training).")
    args = parser.parse_args()
    
    if not os.path.exists(MANIFEST_CSV):
        raise FileNotFoundError(f"{MANIFEST_CSV} not found! Run scripts/03_cut_and_filter.py first.")
        
    manifest_df = pd.read_csv(MANIFEST_CSV)
    print(f"Loaded {len(manifest_df)} clips from {MANIFEST_CSV}")
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32
    
    vae = load_vae(model_name=args.model, device=device, torch_dtype=torch_dtype)
    
    # Default to encoding all clips unless --limit is explicitly passed (AGENTS.md P0.3)
    updated_df = preencode_latents(vae, manifest_df, device=device, torch_dtype=torch_dtype, limit=args.limit)
    
    # Ensure all paths in manifest are POSIX
    updated_df["clip_path"] = updated_df["clip_path"].apply(lambda p: str(p).replace("\\", "/") if pd.notna(p) else p)
    updated_df["latent_path"] = updated_df["latent_path"].apply(lambda p: str(p).replace("\\", "/") if pd.notna(p) else p)
    
    updated_df.to_csv(MANIFEST_CSV, index=False)
    print(f"[3/3] Updated manifest with latent paths saved to {MANIFEST_CSV}")
    
    # Coverage check
    covered = updated_df["latent_path"].notna().sum()
    print(f"Latent Coverage: {covered} / {len(updated_df)} ({covered/len(updated_df)*100:.1f}%)")
    
    if args.assert_100 and covered < len(updated_df):
        raise RuntimeError(
            f"Latent coverage assertion failed: {covered}/{len(updated_df)} clips cached. "
            f"100% coverage is required before training (AGENTS.md P0.3)."
        )


if __name__ == "__main__":
    main()
