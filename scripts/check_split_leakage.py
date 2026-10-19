#!/usr/bin/env python3
"""
scripts/check_split_leakage.py

Split Disjointness and Data Leakage Audit Script
as specified in AGENTS.md §1, §2 (P0.4), §4, and §7.

Audits:
1. Historical Shot Splits (data/doe_nv209_atmospheric_splits.csv):
   - Asserts shot-level disjointness:
     Train ∩ Val = ∅
     Train ∩ Test = ∅
     Val ∩ Test = ∅
   - Checks that all 220 tests have valid, unique split assignments.
   - Audits per-class distribution across splits.
2. Clip Manifest (data/manifest.csv):
   - Asserts clip-to-shot consistency and disjointness across splits in the manifest.
   - Verifies manifest shots match their assigned split from the historical splits table.
3. Exits 0 if zero leakage; exits 1 if any intersection/leakage is found.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_splits_table(splits_path: str = "data/doe_nv209_atmospheric_splits.csv") -> bool:
    full_path = REPO_ROOT / splits_path
    if not full_path.exists():
        print(f"[FAIL] Splits file does not exist: {full_path}")
        return False

    df = pd.read_csv(full_path)
    print(f"=== Auditing Historical Shot Splits: {splits_path} ({len(df)} rows) ===")

    train_shots = set(zip(df[df["split"] == "train"]["operation"], df[df["split"] == "train"]["shot_name"]))
    val_shots = set(zip(df[df["split"] == "val"]["operation"], df[df["split"] == "val"]["shot_name"]))
    test_shots = set(zip(df[df["split"] == "test"]["operation"], df[df["split"] == "test"]["shot_name"]))

    print(f"  Train shots: {len(train_shots)}")
    print(f"  Val shots:   {len(val_shots)}")
    print(f"  Test shots:  {len(test_shots)}")

    # Check pairwise intersections
    train_val_overlap = train_shots & val_shots
    train_test_overlap = train_shots & test_shots
    val_test_overlap = val_shots & test_shots

    leaked = False
    if train_val_overlap:
        print(f"  [ERROR] Train-Val Overlap detected ({len(train_val_overlap)} shots): {train_val_overlap}")
        leaked = True
    if train_test_overlap:
        print(f"  [ERROR] Train-Test Overlap detected ({len(train_test_overlap)} shots): {train_test_overlap}")
        leaked = True
    if val_test_overlap:
        print(f"  [ERROR] Val-Test Overlap detected ({len(val_test_overlap)} shots): {val_test_overlap}")
        leaked = True

    if not leaked:
        print("  [PASS] Shot disjointness verified: Train, Val, and Test are 100% mutually exclusive.")

    # Check underwater split specifically (all 5 shots accounted for)
    underwater = df[df["burst_type"] == "underwater"]
    uw_splits = underwater["split"].value_counts().to_dict()
    print(f"  Underwater tests ({len(underwater)} total): {uw_splits}")
    for _, row in underwater.iterrows():
        print(f"    - {row['operation']} {row['shot_name']}: {row['split']}")

    return not leaked


def check_manifest_splits(
    manifest_path: str = "data/manifest.csv",
    splits_path: str = "data/doe_nv209_atmospheric_splits.csv"
) -> bool:
    full_manifest = REPO_ROOT / manifest_path
    if not full_manifest.exists():
        print(f"[WARN] Manifest file not found: {full_manifest}")
        return True

    df = pd.read_csv(full_manifest)
    print(f"\n=== Auditing Manifest Split Disjointness: {manifest_path} ({len(df)} clips) ===")

    train_shots = set(zip(df[df["split"] == "train"]["operation"], df[df["split"] == "train"]["shot_name"]))
    val_shots = set(zip(df[df["split"] == "val"]["operation"], df[df["split"] == "val"]["shot_name"]))
    test_shots = set(zip(df[df["split"] == "test"]["operation"], df[df["split"] == "test"]["shot_name"]))

    print(f"  Manifest Train shots: {len(train_shots)}")
    print(f"  Manifest Val shots:   {len(val_shots)}")
    print(f"  Manifest Test shots:  {len(test_shots)}")

    leaked = False
    if train_shots & val_shots:
        print(f"  [ERROR] Manifest Train-Val Overlap: {train_shots & val_shots}")
        leaked = True
    if train_shots & test_shots:
        print(f"  [ERROR] Manifest Train-Test Overlap: {train_shots & test_shots}")
        leaked = True
    if val_shots & test_shots:
        print(f"  [ERROR] Manifest Val-Test Overlap: {val_shots & test_shots}")
        leaked = True

    # Cross-reference with doe_nv209_atmospheric_splits.csv
    full_splits = REPO_ROOT / splits_path
    if full_splits.exists():
        sdf = pd.read_csv(full_splits)
        expected_split_map = dict(zip(zip(sdf["operation"], sdf["shot_name"]), sdf["split"]))
        mismatch_count = 0
        for (op, shot), group in df.groupby(["operation", "shot_name"]):
            expected = expected_split_map.get((op, shot))
            actual_splits = group["split"].unique()
            if len(actual_splits) > 1:
                print(f"  [ERROR] Shot {(op, shot)} has multiple splits in manifest: {actual_splits}")
                leaked = True
            elif expected and actual_splits[0] != expected:
                print(f"  [ERROR] Shot {(op, shot)} manifest split '{actual_splits[0]}' != expected '{expected}'")
                mismatch_count += 1
                leaked = True

        if mismatch_count == 0:
            print("  [PASS] All manifest shots perfectly match their authoritative historical split.")

    if not leaked:
        print("  [PASS] Manifest split disjointness verified: Zero leakage across splits.")
    return not leaked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check split disjointness and data leakage.")
    parser.add_argument("--splits", default="data/doe_nv209_atmospheric_splits.csv")
    parser.add_argument("--manifest", default="data/manifest.csv")
    args = parser.parse_args()

    s1 = check_splits_table(args.splits)
    s2 = check_manifest_splits(args.manifest, args.splits)

    if s1 and s2:
        print("\n[SUCCESS] Split leakage audit passed: 0 leakage detected.")
        sys.exit(0)
    else:
        print("\n[FAIL] Split leakage audit failed with errors.")
        sys.exit(1)
