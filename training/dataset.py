"""
training/dataset.py

PyTorch Dataset for Phase 1 (Spatial LoRA) and Phase 2 (Motion LoRA) Training.
Loads either pre-encoded latent tensors (4, 16, H/8, W/8) or raw video frames,
and pairs them with CLIP-tokenized conditioning prompts.
"""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
from transformers import CLIPTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


class NuclearTestDataset(Dataset):
    """
    Dataset supporting both Spatial (frame-level) and Motion (sequence-level) training.
    
    Args:
        manifest_path (str): Path to manifest.csv
        splits_path (str, optional): Path to doe_nv209_atmospheric_splits.csv
        split (str): 'train', 'val', or 'test'
        tokenizer (CLIPTokenizer, optional): CLIP tokenizer
        mode (str): 'spatial' (returns single frame latents) or 'motion' (returns 16-frame sequence latents)
        resolution (int): Target frame resolution (default 512)
        load_latents (bool): If True, load pre-encoded .pt tensors; if False, load raw MP4
    """
    def __init__(
        self,
        manifest_path="data/manifest.csv",
        splits_path="data/doe_nv209_atmospheric_splits.csv",
        split="train",
        tokenizer=None,
        mode="motion",
        resolution=512,
        load_latents=True
    ):
        super().__init__()
        self.mode = mode
        self.resolution = resolution
        self.load_latents = load_latents
        
        full_manifest_path = os.path.join(PROJECT_ROOT, manifest_path) if not os.path.isabs(manifest_path) else manifest_path
        if not os.path.exists(full_manifest_path):
            raise FileNotFoundError(f"Manifest not found: {full_manifest_path}")
            
        df = pd.read_csv(full_manifest_path)
        
        # Filter by split if column exists
        if "split" in df.columns:
            filtered_df = df[df["split"] == split]
            self.df = filtered_df if len(filtered_df) > 0 else df
        else:
            self.df = df
            
        self.df = self.df.reset_index(drop=True)
        
        # Initialize tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer"
            )
        else:
            self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def _load_latent_tensor(self, row):
        """Load pre-encoded .pt latent tensor of shape (4, 16, 64, 64)."""
        latent_path = row.get("latent_path")
        if pd.notna(latent_path):
            full_path = os.path.join(PROJECT_ROOT, str(latent_path))
            if os.path.exists(full_path):
                tensor = torch.load(full_path, map_location="cpu", weights_only=True)
                return tensor
                
        # Try direct file lookup in latents/ directory
        clip_id = row["clip_id"]
        direct_path = os.path.join(PROJECT_ROOT, "latents", f"{clip_id}.pt")
        if os.path.exists(direct_path):
            return torch.load(direct_path, map_location="cpu", weights_only=True)
            
        return None

    def _load_video_frames(self, clip_path):
        """Fallback: Read 16 frames from MP4, normalize to [-1, 1]. Shape: (16, 3, H, W)."""
        full_path = os.path.join(PROJECT_ROOT, clip_path) if not os.path.isabs(clip_path) else clip_path
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            return torch.zeros((16, 3, self.resolution, self.resolution), dtype=torch.float32)
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.resolution, self.resolution))
            norm = (resized.astype(np.float32) / 127.5) - 1.0
            chw = np.transpose(norm, (2, 0, 1))
            frames.append(chw)
        cap.release()
        
        if len(frames) < 16:
            # Pad if needed
            while len(frames) < 16:
                frames.append(frames[-1] if frames else np.zeros((3, self.resolution, self.resolution), dtype=np.float32))
        frames = frames[:16]
        return torch.tensor(np.array(frames), dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row["prompt"]
        
        # Tokenize prompt
        tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokens.input_ids.squeeze(0)
        
        latent_tensor = None
        if self.load_latents:
            latent_tensor = self._load_latent_tensor(row)
            
        if latent_tensor is not None:
            # latent_tensor shape: (4, 16, H/8, W/8)
            if self.mode == "spatial":
                # Sample random frame or middle frame from sequence: (4, H/8, W/8)
                t_idx = np.random.randint(0, latent_tensor.shape[1])
                sample_data = latent_tensor[:, t_idx, :, :]
            else:
                # Full 16-frame sequence for motion LoRA: (4, 16, H/8, W/8)
                sample_data = latent_tensor
            has_latents = True
        else:
            # Raw video frames
            frames = self._load_video_frames(row["clip_path"])
            if self.mode == "spatial":
                t_idx = np.random.randint(0, 16)
                sample_data = frames[t_idx] # (3, H, W)
            else:
                # (16, 3, H, W) -> (3, 16, H, W)
                sample_data = frames.permute(1, 0, 2, 3)
            has_latents = False

        return {
            "data": sample_data,
            "has_latents": has_latents,
            "input_ids": input_ids,
            "prompt": prompt,
            "clip_id": row["clip_id"]
        }
