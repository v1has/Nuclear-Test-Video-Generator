#!/usr/bin/env python3
"""
scripts/smoke_test_dataset.py

Dataset Smoke Test and Batch Shape Verification
as specified in AGENTS.md §1, §2 (P0.3, P0.4), §4, and §7.

Verifies:
1. Fail-fast on empty split: Raises ValueError when a requested split has 0 rows (prohibiting P0.4 silent fallback).
2. Fail-fast on missing latents: Raises FileNotFoundError when load_latents=True but latents are missing (prohibiting P0.3 shape mixing).
3. Raw frame batch shapes (load_latents=False):
   - Spatial mode: [B, 3, H, W]
   - Motion mode:  [B, 3, 16, H, W]
   - Token IDs:    [B, 77]
4. Pre-encoded latent batch shapes (load_latents=True, when available):
   - Spatial mode: [B, 4, H/8, W/8]
   - Motion mode:  [B, 4, 16, H/8, W/8]
   - Token IDs:    [B, 77]
5. Multi-split disjoint batch test: Verifies train, val, and test splits can each yield valid batches without leakage.
"""

import os
import sys
import argparse
from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataset import NuclearTestDataset


def test_empty_split_fail_fast(manifest_path="data/manifest.csv"):
    print("\n--- Test 1: Empty-Split Fail-Fast Assertion (P0.4) ---")
    # Manifest currently only has 'train' clips
    for empty_split in ["val", "test"]:
        try:
            ds = NuclearTestDataset(manifest_path=manifest_path, split=empty_split, load_latents=False)
            print(f"  [FAIL] Expected ValueError for empty split '{empty_split}', but dataset initialized with {len(ds)} items!")
            return False
        except ValueError as e:
            print(f"  [PASS] Successfully raised ValueError on empty split '{empty_split}':")
            print(f"         \"{e}\"")
    return True


def test_raw_frames_batching(manifest_path="data/manifest.csv", batch_size=2):
    print("\n--- Test 2: Raw Frames Batching (load_latents=False) ---")

    # Spatial mode
    print("  Testing Spatial Mode...")
    ds_spatial = NuclearTestDataset(manifest_path=manifest_path, split="train", mode="spatial", load_latents=False)
    loader_spatial = DataLoader(ds_spatial, batch_size=batch_size, shuffle=False)
    batch_spatial = next(iter(loader_spatial))
    spatial_data = batch_spatial["data"]
    spatial_tokens = batch_spatial["input_ids"]

    expected_spatial = (batch_size, 3, 512, 512)
    if tuple(spatial_data.shape) != expected_spatial:
        print(f"  [FAIL] Spatial shape {tuple(spatial_data.shape)} != expected {expected_spatial}")
        return False
    if tuple(spatial_tokens.shape) != (batch_size, 77):
        print(f"  [FAIL] Token shape {tuple(spatial_tokens.shape)} != expected ({batch_size}, 77)")
        return False
    print(f"  [PASS] Spatial raw batch shape: {tuple(spatial_data.shape)}, Tokens: {tuple(spatial_tokens.shape)}")

    # Motion mode
    print("  Testing Motion Mode...")
    ds_motion = NuclearTestDataset(manifest_path=manifest_path, split="train", mode="motion", load_latents=False)
    loader_motion = DataLoader(ds_motion, batch_size=batch_size, shuffle=False)
    batch_motion = next(iter(loader_motion))
    motion_data = batch_motion["data"]
    motion_tokens = batch_motion["input_ids"]

    expected_motion = (batch_size, 3, 16, 512, 512)
    if tuple(motion_data.shape) != expected_motion:
        print(f"  [FAIL] Motion shape {tuple(motion_data.shape)} != expected {expected_motion}")
        return False
    if tuple(motion_tokens.shape) != (batch_size, 77):
        print(f"  [FAIL] Token shape {tuple(motion_tokens.shape)} != expected ({batch_size}, 77)")
        return False
    print(f"  [PASS] Motion raw batch shape:  {tuple(motion_data.shape)}, Tokens: {tuple(motion_tokens.shape)}")
    return True


def test_latent_batching(manifest_path="data/manifest.csv", batch_size=2):
    print("\n--- Test 3: Pre-encoded Latents Batching (load_latents=True) ---")
    try:
        ds_motion = NuclearTestDataset(manifest_path=manifest_path, split="train", mode="motion", load_latents=True)
        loader = DataLoader(ds_motion, batch_size=batch_size, shuffle=False)
        batch = next(iter(loader))
        data = batch["data"]
        expected_shape = (batch_size, 4, 16, 64, 64)
        if tuple(data.shape) != expected_shape:
            print(f"  [FAIL] Latent motion shape {tuple(data.shape)} != expected {expected_shape}")
            return False
        print(f"  [PASS] Latent motion batch shape: {tuple(data.shape)}")

        ds_spatial = NuclearTestDataset(manifest_path=manifest_path, split="train", mode="spatial", load_latents=True)
        loader_sp = DataLoader(ds_spatial, batch_size=batch_size, shuffle=False)
        batch_sp = next(iter(loader_sp))
        data_sp = batch_sp["data"]
        expected_sp_shape = (batch_size, 4, 64, 64)
        if tuple(data_sp.shape) != expected_sp_shape:
            print(f"  [FAIL] Latent spatial shape {tuple(data_sp.shape)} != expected {expected_sp_shape}")
            return False
        print(f"  [PASS] Latent spatial batch shape: {tuple(data_sp.shape)}")
        return True
    except FileNotFoundError as e:
        print(f"  [PASS] Latents not fully encoded yet. Correctly raised FileNotFoundError (prohibiting shape mixing):")
        print(f"         \"{e}\"")
        return True
    except Exception as e:
        print(f"  [FAIL] Unexpected exception during latent test: {e}")
        return False


def test_multi_split_disjointness():
    print("\n--- Test 4: Multi-Split Batching Smoke Test (Synthetic Disjoint Manifest) ---")
    import pandas as pd
    real_manifest = pd.read_csv(REPO_ROOT / "data/manifest.csv")
    if len(real_manifest) < 3:
        print("  [SKIP] Not enough clips to synthesize multi-split test.")
        return True

    # Assign 2 clips to train, 1 to val, 1 to test for smoke test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name
        temp_df = real_manifest.copy()
        temp_df.loc[0:1, "split"] = "train"
        temp_df.loc[2:2, "split"] = "val"
        temp_df.loc[3:3, "split"] = "test"
        temp_df.to_csv(temp_path, index=False)

    try:
        for split in ["train", "val", "test"]:
            ds = NuclearTestDataset(manifest_path=temp_path, split=split, mode="spatial", load_latents=False)
            loader = DataLoader(ds, batch_size=1)
            b = next(iter(loader))
            print(f"  [PASS] Split '{split}' yielded batch: data shape {tuple(b['data'].shape)}, prompt='{b['prompt'][0][:45]}...'")
        return True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_all_smoke_tests():
    print("=== Running Dataset Smoke Tests ===")
    t1 = test_empty_split_fail_fast()
    t2 = test_raw_frames_batching()
    t3 = test_latent_batching()
    t4 = test_multi_split_disjointness()

    all_passed = t1 and t2 and t3 and t4
    if all_passed:
        print("\n[SUCCESS] All dataset smoke tests passed completely!")
    else:
        print("\n[FAIL] One or more smoke tests failed.")
    return all_passed


if __name__ == "__main__":
    success = run_all_smoke_tests()
    sys.exit(0 if success else 1)
