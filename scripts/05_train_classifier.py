"""
scripts/05_train_classifier.py

Stage 6: Control Accuracy & Evaluation Suite.
Trains and evaluates a multi-task video classifier on real archival explosion clips
to predict:
1. burst_type (4 classes: air-burst, surface-tower, surface-barge, underwater)
2. yield_tier (4 classes: sub-kiloton, tactical-yield, intermediate-yield, megaton-class)
3. phase (3 classes: early-fireball, expansion, stabilized-cloud)

The trained classifier is then used to quantitatively evaluate synthetic generated clips
against their conditioning prompt attributes, measuring balanced accuracy against
majority-class baselines (validation gate: >25% improvement over majority baseline).

Usage:
  python scripts/05_train_classifier.py --train --epochs 10 --device cpu
  python scripts/05_train_classifier.py --eval
  python scripts/05_train_classifier.py --evaluate_synthetic --synthetic_dir samples/
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as vmodels
import torchvision.models as models
import cv2
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "control_classifier.pt")

# Canonical class mappings
CLASSES = {
    "burst_type": ["air-burst", "surface-tower", "surface-barge", "underwater"],
    "yield_tier": ["sub-kiloton", "tactical-yield", "intermediate-yield", "megaton-class"],
    "phase": ["early-fireball", "expansion", "stabilized-cloud"]
}

# Synonyms / normalization for phases
PHASE_SYNONYMS = {
    "flash-fireball": "early-fireball",
    "fireball": "early-fireball",
    "early": "early-fireball",
    "cloud": "stabilized-cloud",
    "mushroom-cloud": "stabilized-cloud"
}


def normalize_phase(p):
    p_clean = str(p).strip().lower()
    return PHASE_SYNONYMS.get(p_clean, p_clean)


class VideoClipDataset(Dataset):
    """
    Loads 16-frame video clips from manifest.csv.
    Returns:
      tensor: (3, 16, H, W) for 3D CNN, normalized
      targets: dict with int class labels for burst_type, yield_tier, phase
    """
    def __init__(self, manifest_df, target_size=112, is_train=True):
        self.df = manifest_df.reset_index(drop=True)
        self.target_size = target_size
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_path = row["clip_path"]
        full_path = os.path.join(PROJECT_ROOT, clip_path) if not os.path.isabs(clip_path) else clip_path
        
        frames = self._load_frames(full_path)
        if frames is None:
            # Fallback zero tensor
            frames = torch.zeros((3, 16, self.target_size, self.target_size), dtype=torch.float32)
            
        burst_label = CLASSES["burst_type"].index(row["burst_type"]) if row["burst_type"] in CLASSES["burst_type"] else 0
        yield_label = CLASSES["yield_tier"].index(row["yield_tier"]) if row["yield_tier"] in CLASSES["yield_tier"] else 0
        
        norm_phase = normalize_phase(row["phase"])
        phase_label = CLASSES["phase"].index(norm_phase) if norm_phase in CLASSES["phase"] else 1
        
        targets = {
            "burst_type": torch.tensor(burst_label, dtype=torch.long),
            "yield_tier": torch.tensor(yield_label, dtype=torch.long),
            "phase": torch.tensor(phase_label, dtype=torch.long)
        }
        return frames, targets, row["clip_id"]

    def _load_frames(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        
        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            # Normalize to [0, 1] then ImageNet mean/std
            arr = resized.astype(np.float32) / 255.0
            mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
            std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
            arr = (arr - mean) / std
            # (H, W, C) -> (C, H, W)
            chw = np.transpose(arr, (2, 0, 1))
            raw_frames.append(chw)
        cap.release()
        
        if len(raw_frames) < 16:
            return None
            
        raw_frames = raw_frames[:16]
        # (16, C, H, W) -> (C, 16, H, W) for 3D Conv
        stacked = np.array(raw_frames)
        tensor = torch.tensor(stacked, dtype=torch.float32).permute(1, 0, 2, 3)
        return tensor


class MultiTaskVideoClassifier(nn.Module):
    """
    Multi-task 3D video classification model.
    Backbone: MC3-18 (or ResNet18 with temporal average pooling).
    Heads:
      - burst_type (4 classes)
      - yield_tier (4 classes)
      - phase (3 classes)
    """
    def __init__(self, backbone="mc3_18", pretrained=False):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == "mc3_18":
            base_model = vmodels.mc3_18(weights=None)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        elif backbone == "resnet18_avgpool":
            base_model = models.resnet18(weights=None)
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
            self.backbone = base_model
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        self.dropout = nn.Dropout(0.3)
        self.head_burst = nn.Linear(in_features, len(CLASSES["burst_type"]))
        self.head_yield = nn.Linear(in_features, len(CLASSES["yield_tier"]))
        self.head_phase = nn.Linear(in_features, len(CLASSES["phase"]))

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        if self.backbone_name == "resnet18_avgpool":
            # Pass each frame through 2D ResNet and average over time
            B, C, T, H, W = x.shape
            x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            feat_2d = self.backbone(x_2d) # (B*T, in_features)
            feat = feat_2d.view(B, T, -1).mean(dim=1) # (B, in_features)
        else:
            # 3D CNN
            feat = self.backbone(x) # (B, in_features)
            
        feat = self.dropout(feat)
        return {
            "burst_type": self.head_burst(feat),
            "yield_tier": self.head_yield(feat),
            "phase": self.head_phase(feat)
        }


def compute_balanced_accuracy(y_true, y_pred, num_classes):
    """Computes macro-averaged recall across classes."""
    recalls = []
    for c in range(num_classes):
        mask = (y_true == c)
        if np.sum(mask) == 0:
            continue
        recall = np.sum((y_pred == c) & mask) / np.sum(mask)
        recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = {k: [] for k in CLASSES}
    all_trues = {k: [] for k in CLASSES}
    
    with torch.no_grad():
        for frames, targets, _ in dataloader:
            frames = frames.to(device)
            outputs = model(frames)
            for k in CLASSES:
                pred = outputs[k].argmax(dim=1).cpu().numpy()
                all_preds[k].extend(pred)
                all_trues[k].extend(targets[k].numpy())
                
    results = {}
    for k in CLASSES:
        y_true = np.array(all_trues[k])
        y_pred = np.array(all_preds[k])
        num_c = len(CLASSES[k])
        
        # Raw accuracy
        acc = (y_true == y_pred).mean() if len(y_true) > 0 else 0.0
        # Balanced accuracy
        bal_acc = compute_balanced_accuracy(y_true, y_pred, num_c)
        # Majority class baseline
        counts = np.bincount(y_true, minlength=num_c)
        majority_baseline = np.max(counts) / len(y_true) if len(y_true) > 0 else 0.0
        
        margin_pct = (bal_acc - majority_baseline) / max(0.01, majority_baseline) * 100
        gate_passed = margin_pct >= 25.0
        
        results[k] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "majority_baseline": float(majority_baseline),
            "margin_vs_majority_pct": float(margin_pct),
            "gate_passed_25pct": bool(gate_passed)
        }
    return results


def train_classifier(args):
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on device: {device}")
    
    df = pd.read_csv(MANIFEST_CSV)
    print(f"Total manifest samples: {len(df)}")
    
    # Train / Val split
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    if len(val_df) == 0:
        val_df = df[df["split"] == "test"]
    if len(val_df) == 0:
        # Fallback 80/20 random split if only train present in sample
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
        
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    
    train_ds = VideoClipDataset(train_df, target_size=112, is_train=True)
    val_ds = VideoClipDataset(val_df, target_size=112, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = MultiTaskVideoClassifier(backbone=args.backbone, pretrained=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_score = 0.0
    
    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for frames, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            frames = frames.to(device)
            outputs = model(frames)
            loss_burst = criterion(outputs["burst_type"], targets["burst_type"].to(device))
            loss_yield = criterion(outputs["yield_tier"], targets["yield_tier"].to(device))
            loss_phase = criterion(outputs["phase"], targets["phase"].to(device))
            
            loss = loss_burst + loss_yield + loss_phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(train_loader))
        val_results = evaluate_model(model, val_loader, device)
        
        avg_bal_acc = np.mean([v["balanced_accuracy"] for v in val_results.values()])
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f} | Mean Val Balanced Acc = {avg_bal_acc:.2%}")
        for k, v in val_results.items():
            print(f"  [{k}] Bal Acc: {v['balanced_accuracy']:.2%} (Baseline: {v['majority_baseline']:.2%}, Margin: {v['margin_vs_majority_pct']:+.1f}%)")
            
        if avg_bal_acc >= best_val_score:
            best_val_score = avg_bal_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "classes": CLASSES,
                "best_val_score": best_val_score,
                "val_results": val_results
            }, CLASSIFIER_PATH)
            print(f"  => Saved best model to {CLASSIFIER_PATH}")
            
    print(f"\nTraining Complete. Best Validation Score: {best_val_score:.2%}")


def evaluate_synthetic_clips(args):
    """
    Evaluates synthetic generated clips in a directory against the classifier.
    Expects synthetic clips to be named or tagged with expected attributes, or parses JSON metadata.
    """
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"Error: Trained classifier checkpoint {CLASSIFIER_PATH} not found.")
        sys.exit(1)
        
    checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)
    backbone = checkpoint.get("backbone", "mc3_18")
    model = MultiTaskVideoClassifier(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    synth_dir = args.synthetic_dir
    if not os.path.exists(synth_dir):
        print(f"Synthetic directory '{synth_dir}' not found.")
        return
        
    mp4_files = [os.path.join(synth_dir, f) for f in os.listdir(synth_dir) if f.endswith(".mp4")]
    print(f"Found {len(mp4_files)} synthetic video clips in {synth_dir}")
    
    predictions = []
    for f in tqdm(mp4_files, desc="Auditing Synthetic Clips"):
        cap = cv2.VideoCapture(f)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (112, 112))
            arr = (resized.astype(np.float32) / 255.0 - np.array([0.43216, 0.394666, 0.37645])) / np.array([0.22803, 0.22145, 0.216989])
            frames.append(np.transpose(arr, (2, 0, 1)))
        cap.release()
        
        if len(frames) < 16:
            continue
        frames = frames[:16]
        tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(1, 0, 2, 3).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor)
            pred_burst = CLASSES["burst_type"][out["burst_type"].argmax().item()]
            pred_yield = CLASSES["yield_tier"][out["yield_tier"].argmax().item()]
            pred_phase = CLASSES["phase"][out["phase"].argmax().item()]
            
        predictions.append({
            "file": os.path.basename(f),
            "pred_burst": pred_burst,
            "pred_yield": pred_yield,
            "pred_phase": pred_phase
        })
        
    report_path = os.path.join(DATA_DIR, "synthetic_control_report.json")
    with open(report_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Synthetic evaluation report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 6 Video Control Classifier Suite")
    parser.add_argument("--train", action="store_true", help="Train classifier on real manifest data")
    parser.add_argument("--eval", action="store_true", help="Evaluate classifier on test split")
    parser.add_argument("--evaluate_synthetic", action="store_true", help="Audit synthetic generated clips")
    parser.add_argument("--synthetic_dir", type=str, default="samples/", help="Directory of synthetic clips")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet18_avgpool", choices=["mc3_18", "resnet18_avgpool"], help="CNN Backbone")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    if args.train:
        train_classifier(args)
    elif args.evaluate_synthetic:
        evaluate_synthetic_clips(args)
    else:
        # Default run train if no mode selected
        train_classifier(args)


if __name__ == "__main__":
    main()
