#!/usr/bin/env python3
"""
scripts/validate_manifest.py

Manifest schema, path formatting, and integrity validation script
as specified in AGENTS.md §1, §2 (P0.5), §4, and §7.

Checks:
1. Every prompt token validates against schema.py.
2. Extracted prompt attributes match row columns (burst_type, yield_tier, terrain, lighting, phase, film_look, smoke_trails).
3. All file paths use POSIX forward slashes ('/'), never Windows backslashes ('\\').
4. Validates split membership ('train', 'val', 'test').
5. Verifies clip_path exists on disk.
6. Verifies latent_path formatting, existence, and tensor shapes if cached.
7. Prints detailed audit statistics. Exits 0 on success, 1 on failure.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import schema


def validate_manifest(manifest_path: str = "data/manifest.csv", require_latents: bool = False, enforce_gate: bool = False) -> bool:
    full_path = REPO_ROOT / manifest_path
    if not full_path.exists():
        print(f"[FAIL] Manifest file does not exist: {full_path}")
        return False

    df = pd.read_csv(full_path)
    total_rows = len(df)
    print(f"=== Validating Manifest: {manifest_path} ({total_rows} rows) ===")

    errors = []
    warnings = []

    required_cols = [
        "clip_id", "operation", "shot_name", "split", "burst_type",
        "yield_tier", "terrain", "lighting", "phase", "film_look",
        "prompt", "clip_path"
    ]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required manifest column: '{col}'")

    if errors:
        for err in errors:
            print(f"  [ERROR] {err}")
        return False

    # Row-by-row validation
    num_valid_prompts = 0
    posix_path_violations = 0
    missing_clip_files = 0
    cached_latents = 0
    missing_latent_files = 0
    mismatched_tokens = 0

    for idx, row in df.iterrows():
        clip_id = row.get("clip_id", f"row_{idx}")

        # 1. Split check
        split = row.get("split")
        if split not in ["train", "val", "test"]:
            errors.append(f"Row {idx} ({clip_id}): Invalid split '{split}'. Must be train, val, or test.")

        # 2. Schema check on row columns
        try:
            schema.validate_manifest_row(row.to_dict())
        except Exception as e:
            errors.append(f"Row {idx} ({clip_id}): Column validation failed: {e}")

        # 3. Prompt validation against schema
        prompt = str(row.get("prompt", ""))
        try:
            tokens = schema.validate_prompt(prompt)
            num_valid_prompts += 1

            # Check that prompt matches row columns
            for attr in ["burst_type", "yield_tier", "terrain", "lighting", "phase", "film_look"]:
                row_val = str(row.get(attr, ""))
                token_val = str(tokens.get(attr, ""))
                if row_val != token_val:
                    errors.append(
                        f"Row {idx} ({clip_id}): Mismatch on {attr}: prompt has '{token_val}', row has '{row_val}'"
                    )
                    mismatched_tokens += 1

            if "smoke_trails" in row:
                row_smoke = bool(row["smoke_trails"])
                token_smoke = bool(tokens.get("smoke_trails"))
                if row_smoke != token_smoke:
                    errors.append(
                        f"Row {idx} ({clip_id}): Mismatch on smoke_trails: prompt has '{token_smoke}', row has '{row_smoke}'"
                    )
                    mismatched_tokens += 1
        except Exception as e:
            errors.append(f"Row {idx} ({clip_id}): Invalid prompt '{prompt}': {e}")

        # 4. Path POSIX check
        clip_path = str(row.get("clip_path", ""))
        if "\\" in clip_path:
            errors.append(f"Row {idx} ({clip_id}): clip_path contains Windows backslash: '{clip_path}'")
            posix_path_violations += 1

        full_clip_path = REPO_ROOT / clip_path
        if not full_clip_path.exists():
            errors.append(f"Row {idx} ({clip_id}): Video file not found at: {clip_path}")
            missing_clip_files += 1

        # 5. Latent check
        latent_path = row.get("latent_path")
        if pd.notna(latent_path) and str(latent_path).strip():
            lp_str = str(latent_path).strip()
            if "\\" in lp_str:
                errors.append(f"Row {idx} ({clip_id}): latent_path contains Windows backslash: '{lp_str}'")
                posix_path_violations += 1
            full_lp = REPO_ROOT / lp_str
            if full_lp.exists():
                cached_latents += 1
                try:
                    t = torch.load(full_lp, map_location="cpu", weights_only=True)
                    # Check shape: expect (4, 16, H, W)
                    if t.dim() != 4 or t.shape[0] != 4 or t.shape[1] != 16:
                        errors.append(
                            f"Row {idx} ({clip_id}): Latent tensor has unexpected shape {tuple(t.shape)}, expected (4, 16, 64, 64)"
                        )
                except Exception as e:
                    errors.append(f"Row {idx} ({clip_id}): Corrupted latent tensor: {e}")
            else:
                missing_latent_files += 1
                if require_latents:
                    errors.append(f"Row {idx} ({clip_id}): Latent file specified but missing: {lp_str}")
        else:
            if require_latents:
                errors.append(f"Row {idx} ({clip_id}): Missing latent_path in manifest.")

    # Audit Summaries
    print("\n--- Summary Audit ---")
    print(f"Total Clips:               {total_rows}")
    print(f"Distinct Operations:       {df['operation'].nunique()}")
    print(f"Distinct Shots:            {df.groupby(['operation', 'shot_name']).ngroups}")
    print(f"Splits Breakdown:          {df['split'].value_counts().to_dict()}")
    print(f"Burst Types:               {df['burst_type'].value_counts().to_dict()}")
    print(f"Yield Tiers:               {df['yield_tier'].value_counts().to_dict()}")
    print(f"Phases:                    {df['phase'].value_counts().to_dict()}")
    print(f"Valid Prompts (Schema):    {num_valid_prompts} / {total_rows}")
    print(f"Cached Latents (.pt):      {cached_latents} / {total_rows} ({cached_latents/max(1, total_rows):.1%})")

    distinct_shots = df.groupby(['operation', 'shot_name']).ngroups
    split_counts = df['split'].value_counts().to_dict()
    val_count = split_counts.get('val', 0)
    test_count = split_counts.get('test', 0)
    latent_coverage = cached_latents / max(1, total_rows)

    # Data Readiness Gate (AGENTS.md §4)
    gate_failures = []
    if distinct_shots < 30:
        gate_failures.append(f"Distinct shots: {distinct_shots} / 30 required (AGENTS.md P0.1)")
    if val_count == 0:
        gate_failures.append("Validation split has 0 clips (AGENTS.md P0.4)")
    if test_count == 0:
        gate_failures.append("Test split has 0 clips (AGENTS.md P0.4)")
    if cached_latents < total_rows:
        gate_failures.append(f"Latent coverage: {cached_latents}/{total_rows} ({latent_coverage:.1%}) < 100% (AGENTS.md P0.3)")

    gate_passed = len(gate_failures) == 0

    print("\n--- Training Readiness Gate Audit (AGENTS.md §4) ---")
    if gate_passed:
        print("  [DATA GATE: PASSED] Ready for Phase 1/2 LoRA training.")
    else:
        print("  [DATA GATE: FAILED - Phase 1/2 LoRA training must remain BLOCKED]")
        for gf in gate_failures:
            print(f"    x {gf}")

    if errors:
        print(f"\n[FAIL] {len(errors)} validation error(s) detected:")
        for err in errors[:25]:
            print(f"  x {err}")
        if len(errors) > 25:
            print(f"  ... and {len(errors) - 25} more errors.")
        return False

    if enforce_gate and not gate_passed:
        print("\n[FAIL] Manifest failed training readiness gate checks (--gate enforced).")
        return False

    print("\n[SUCCESS] Manifest schema & file integrity validated.")
    print("  + All prompt tokens strictly conform to schema.py")
    print("  + All paths use POSIX '/' formatting")
    print("  + Video files exist and match manifest entries")
    if not gate_passed:
        print("  ! NOTE: Data gate remains FAILED pending remote multi-reel download (1 shot currently).")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate manifest schema and integrity.")
    parser.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--require-latents", action="store_true", help="Enforce 100% latent coverage")
    parser.add_argument("--gate", action="store_true", help="Enforce Stage 4 data readiness gate (>=30 shots, non-empty val/test, 100% latents)")
    args = parser.parse_args()

    success = validate_manifest(args.manifest, require_latents=args.require_latents, enforce_gate=args.gate)
    sys.exit(0 if success else 1)
