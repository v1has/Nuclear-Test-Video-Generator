"""
training/dataset.py

PyTorch Dataset for Phase 1 (Spatial LoRA) and Phase 2 (Motion LoRA) Training.
Loads either pre-encoded latent tensors (4, 16, H/8, W/8) or raw video frames,
and pairs them with CLIP-tokenized conditioning prompts.

Fixes P0.3, P0.4, and P1 items:
- Fail-fast: No empty-split fallback (raises ValueError instead of returning df)
- No tensor shape mixing: Strictly enforces all items in a batch are either latents
  or raw frames; raises FileNotFoundError on missing latents when load_latents=True
- Uses splits_path if manifest lacks split column
- Validates prompt tokens against schema.py
- Normalizes all paths to POSIX '/'
"""

import os
import sys
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
from transformers import CLIPTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import schema


class NuclearTestDataset(Dataset):
    """
    Dataset supporting both Spatial (frame-level) and Motion (sequence-level) training.
    """
    def __init__(
        self,
        manifest_path="data/manifest.csv",
        splits_path="data/doe_nv209_atmospheric_splits.csv",
        split="train",
        tokenizer=None,
        mode="motion",
        resolution=512,
        load_latents=True,
        validate_schema=True
    ):
        super().__init__()
        if mode not in ["spatial", "motion"]:
            raise ValueError(f"mode must be 'spatial' or 'motion', got '{mode}'")
        self.mode = mode
        self.resolution = resolution
        self.load_latents = load_latents
        self.split = split

        full_manifest_path = os.path.join(REPO_ROOT, manifest_path) if not os.path.isabs(manifest_path) else manifest_path
        if not os.path.exists(full_manifest_path):
            raise FileNotFoundError(f"Manifest not found: {full_manifest_path}")

        df = pd.read_csv(full_manifest_path)

        # Merge split from splits_path if missing in manifest
        if "split" not in df.columns and splits_path:
            full_splits_path = os.path.join(REPO_ROOT, splits_path) if not os.path.isabs(splits_path) else splits_path
            if os.path.exists(full_splits_path):
                splits_df = pd.read_csv(full_splits_path)
                shot_to_split = dict(zip(
                    zip(splits_df["operation"], splits_df["shot_name"]),
                    splits_df["split"]
                ))
                df["split"] = df.apply(
                    lambda r: shot_to_split.get((r.get("operation"), r.get("shot_name")), "unassigned"),
                    axis=1
                )

        # Filter strictly by split
        if "split" in df.columns and split is not None:
            filtered_df = df[df["split"] == split].copy()
            if len(filtered_df) == 0:
                available = df["split"].value_counts().to_dict() if "split" in df.columns else {}
                raise ValueError(
                    f"Split '{split}' has 0 rows in {manifest_path}. "
                    f"Available splits: {available}. "
                    f"Silent fallback to entire dataset is strictly forbidden (AGENTS.md P0.4)."
                )
            self.df = filtered_df
        else:
            self.df = df.copy()

        self.df = self.df.reset_index(drop=True)

        # Initialize tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer"
            )
        else:
            self.tokenizer = tokenizer

        # Assert latent coverage if load_latents=True
        if self.load_latents:
            missing_latents = []
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                lp = row.get("latent_path")
                if pd.isna(lp) or not os.path.exists(os.path.join(REPO_ROOT, str(lp))):
                    direct_path = os.path.join(REPO_ROOT, "latents", f"{row['clip_id']}.pt")
                    if not os.path.exists(direct_path):
                        missing_latents.append(row["clip_id"])
            if missing_latents:
                raise FileNotFoundError(
                    f"Split '{split}' contains {len(missing_latents)} clips without cached latents "
                    f"(e.g. {missing_latents[:3]}). Mixing raw frames with latents violates P0.3. "
                    f"Run scripts/04_preencode_latents.py or set load_latents=False."
                )

        if validate_schema:
            for idx in range(min(10, len(self.df))):
                row = self.df.iloc[idx]
                if "prompt" in row and pd.notna(row["prompt"]):
                    schema.validate_prompt(row["prompt"])

    def __len__(self):
        return len(self.df)

    def _load_latent_tensor(self, row):
        """Load pre-encoded .pt latent tensor of shape (4, 16, 64, 64)."""
        latent_path = row.get("latent_path")
        if pd.notna(latent_path):
            full_path = os.path.join(REPO_ROOT, str(latent_path).replace("\\", "/"))
            if os.path.exists(full_path):
                return torch.load(full_path, map_location="cpu", weights_only=True)

        clip_id = row["clip_id"]
        direct_path = os.path.join(REPO_ROOT, "latents", f"{clip_id}.pt")
        if os.path.exists(direct_path):
            return torch.load(direct_path, map_location="cpu", weights_only=True)

        raise FileNotFoundError(f"Latent file not found for clip {clip_id}")

    def _load_video_frames(self, clip_path):
        """Read 16 frames from MP4, normalize to [-1, 1]. Shape: (16, 3, H, W)."""
        posix_path = str(clip_path).replace("\\", "/")
        full_path = os.path.join(REPO_ROOT, posix_path) if not os.path.isabs(posix_path) else posix_path
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video clip: {full_path}")

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
            while len(frames) < 16:
                frames.append(frames[-1] if frames else np.zeros((3, self.resolution, self.resolution), dtype=np.float32))
        frames = frames[:16]
        return torch.tensor(np.array(frames), dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = row["prompt"]

        tokens = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokens.input_ids.squeeze(0)

        if self.load_latents:
            latent_tensor = self._load_latent_tensor(row)
            # latent_tensor shape: (4, 16, H/8, W/8)
            if self.mode == "spatial":
                t_idx = np.random.randint(0, latent_tensor.shape[1])
                sample_data = latent_tensor[:, t_idx, :, :] # (4, H/8, W/8)
            else:
                sample_data = latent_tensor # (4, 16, H/8, W/8)
            has_latents = True
        else:
            frames = self._load_video_frames(row["clip_path"])
            if self.mode == "spatial":
                t_idx = np.random.randint(0, 16)
                sample_data = frames[t_idx] # (3, H, W)
            else:
                sample_data = frames.permute(1, 0, 2, 3) # (3, 16, H, W)
            has_latents = False

        return {
            "data": sample_data,
            "has_latents": has_latents,
            "input_ids": input_ids,
            "prompt": prompt,
            "clip_id": row["clip_id"]
        }
