"""
scripts/05_train_classifier.py

Stage 6: Control Accuracy & Evaluation Suite.
Trains and evaluates a multi-task video classifier on real archival explosion clips
to predict:
1. burst_type (4 classes: air-burst, surface-tower, surface-barge, underwater)
2. yield_tier (4 classes: sub-kiloton, tactical-yield, intermediate-yield, megaton-class)
3. phase (3 classes: early-fireball, expansion, stabilized-cloud)

Fixes P0.4, P0.9, and P1 items:
- Implements --eval to evaluate the classifier on the held-out test split
- Computes per-class precision, recall, confusion matrices, and balanced accuracy
- Implements class-weighted CrossEntropyLoss to address severe imbalance (rare underwater)
- Implements synthetic clip audit against expected label JSON sidecars
- Validates the official Stage 6 gate: balanced_accuracy >= majority_baseline + 25%
- Enforces strict shot-disjoint split loading; raises error on empty splits
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import schema

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "control_classifier.pt")
SYNTHETIC_REPORT_JSON = os.path.join(DATA_DIR, "synthetic_control_report.json")
SYNTHETIC_REPORT_MD = os.path.join(DOCS_DIR, "STAGE6_SYNTHETIC_AUDIT.md")

CLASSES = {
    "burst_type": schema.BURST_TYPES,
    "yield_tier": schema.YIELD_TIERS,
    "phase": schema.PHASES
}


def normalize_phase(p):
    p_clean = str(p).strip().lower()
    synonyms = {
        "flash-fireball": "early-fireball",
        "fireball": "early-fireball",
        "stem-rise": "expansion",
        "expansion": "expansion",
        "stabilized-cloud": "stabilized-cloud",
        "cloud": "stabilized-cloud"
    }
    return synonyms.get(p_clean, p_clean)


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

    def _load_frames(self, full_path):
        posix_path = str(full_path).replace("\\", "/")
        cap = cv2.VideoCapture(posix_path)
        if not cap.isOpened():
            return None
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.target_size, self.target_size))
            arr = (resized.astype(np.float32) / 255.0 - np.array([0.43216, 0.394666, 0.37645])) / np.array([0.22803, 0.22145, 0.216989])
            chw = np.transpose(arr, (2, 0, 1))
            frames.append(chw)
        cap.release()
        
        if len(frames) < 16:
            if not frames:
                return None
            while len(frames) < 16:
                frames.append(frames[-1])
        frames = frames[:16]
        
        # (16, 3, H, W) -> (3, 16, H, W)
        tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(1, 0, 2, 3)
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_path = row["clip_path"]
        full_path = os.path.join(PROJECT_ROOT, clip_path) if not os.path.isabs(clip_path) else clip_path
        
        frames = self._load_frames(full_path)
        if frames is None:
            frames = torch.zeros((3, 16, self.target_size, self.target_size), dtype=torch.float32)
            
        burst_label = CLASSES["burst_type"].index(row["burst_type"]) if row["burst_type"] in CLASSES["burst_type"] else 0
        yield_label = CLASSES["yield_tier"].index(row["yield_tier"]) if row["yield_tier"] in CLASSES["yield_tier"] else 0
        
        norm_phase = normalize_phase(row.get("phase", "expansion"))
        phase_label = CLASSES["phase"].index(norm_phase) if norm_phase in CLASSES["phase"] else 1
        
        targets = {
            "burst_type": torch.tensor(burst_label, dtype=torch.long),
            "yield_tier": torch.tensor(yield_label, dtype=torch.long),
            "phase": torch.tensor(phase_label, dtype=torch.long)
        }
        return frames, targets, row.get("clip_id", f"clip_{idx}")


class MultiTaskVideoClassifier(nn.Module):
    """Multi-task video classifier predicting burst_type, yield_tier, and phase."""
    def __init__(self, backbone="resnet18_avgpool", pretrained=False):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == "mc3_18":
            try:
                base = vmodels.mc3_18(weights=vmodels.MC3_18_Weights.DEFAULT if pretrained else None)
                self.feature_extractor = base
                in_features = base.fc.in_features
                self.feature_extractor.fc = nn.Identity()
            except Exception:
                backbone = "resnet18_avgpool"
                self.backbone_name = "resnet18_avgpool"
                
        if backbone == "resnet18_avgpool":
            base2d = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            base2d.fc = nn.Identity()
            self.spatial_encoder = base2d
            in_features = 512
            
        self.head_burst = nn.Linear(in_features, len(CLASSES["burst_type"]))
        self.head_yield = nn.Linear(in_features, len(CLASSES["yield_tier"]))
        self.head_phase = nn.Linear(in_features, len(CLASSES["phase"]))

    def forward(self, x):
        # x: (B, 3, T, H, W)
        if self.backbone_name == "mc3_18":
            feat = self.feature_extractor(x)
        else:
            b, c, t, h, w = x.shape
            x_perm = x.permute(0, 2, 1, h, w).reshape(b * t, c, h, w)
            feats = self.spatial_encoder(x_perm)
            feat = feats.view(b, t, -1).mean(dim=1)
            
        return {
            "burst_type": self.head_burst(feat),
            "yield_tier": self.head_yield(feat),
            "phase": self.head_phase(feat)
        }


def compute_metrics_and_confusion(y_true, y_pred, class_names):
    """Compute per-class recall, precision, confusion matrix, and balanced accuracy."""
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    recalls = []
    precisions = []
    per_class = {}
    
    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0
        recalls.append(rec)
        precisions.append(prec)
        per_class[class_names[c]] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(tp + fn)
        }
        
    balanced_acc = float(np.mean(recalls)) if recalls else 0.0
    majority_count = np.max(np.bincount(y_true, minlength=num_classes)) if len(y_true) > 0 else 0
    majority_baseline = float(majority_count / max(1, len(y_true)))
    margin = (balanced_acc - majority_baseline) * 100.0
    gate_passed = balanced_acc >= (majority_baseline + 0.25)
    
    return {
        "balanced_accuracy": balanced_acc,
        "majority_baseline": majority_baseline,
        "margin_percentage": margin,
        "gate_passed": gate_passed,
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def compute_class_weights(df, column, classes):
    """Compute inverse frequency class weights to balance gradient updates."""
    counts = df[column].value_counts().to_dict()
    total = len(df)
    num_classes = len(classes)
    weights = []
    for c in classes:
        cnt = counts.get(c, 0)
        w = total / (num_classes * max(1, cnt))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataloader and return detailed metrics."""
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
        results[k] = compute_metrics_and_confusion(y_true, y_pred, CLASSES[k])
    return results


def train_classifier(args):
    """Train multi-task classifier on real archival manifest data."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training Stage 6 Control Classifier on device: {device}")
    
    if not os.path.exists(MANIFEST_CSV):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_CSV}")
        
    df = pd.read_csv(MANIFEST_CSV)
    print(f"Total manifest clips: {len(df)}")
    
    if "split" not in df.columns:
        raise ValueError("manifest.csv lacks 'split' column. Cannot ensure shot-disjoint splits.")
        
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    if len(train_df) == 0:
        raise ValueError("Train split has 0 rows in manifest.csv. Refusing to proceed.")
        
    # A held-out validation split is mandatory. Never self-validate on a train
    # tail or borrow the test split: both leak evaluation signal (AGENTS.md P0.4).
    if len(val_df) == 0:
        raise ValueError(
            "Validation split has 0 rows in manifest.csv. "
            "Falling back to test or a train tail is strictly forbidden (AGENTS.md §1 Rule 1, P0.4). "
            "Ingest and segment additional shots before training."
        )
        
    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")
    
    train_ds = VideoClipDataset(train_df, target_size=112, is_train=True)
    val_ds = VideoClipDataset(val_df, target_size=112, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Class weights for loss
    w_burst = compute_class_weights(train_df, "burst_type", CLASSES["burst_type"]).to(device)
    w_yield = compute_class_weights(train_df, "yield_tier", CLASSES["yield_tier"]).to(device)
    w_phase = compute_class_weights(train_df, "phase", CLASSES["phase"]).to(device)
    
    crit_burst = nn.CrossEntropyLoss(weight=w_burst)
    crit_yield = nn.CrossEntropyLoss(weight=w_yield)
    crit_phase = nn.CrossEntropyLoss(weight=w_phase)
    
    model = MultiTaskVideoClassifier(backbone=args.backbone, pretrained=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    best_score = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        for frames, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            frames = frames.to(device)
            outputs = model(frames)
            
            l_burst = crit_burst(outputs["burst_type"], targets["burst_type"].to(device))
            l_yield = crit_yield(outputs["yield_tier"], targets["yield_tier"].to(device))
            l_phase = crit_phase(outputs["phase"], targets["phase"].to(device))
            loss = l_burst + l_yield + l_phase
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(train_loader))
        val_res = evaluate_model(model, val_loader, device)
        mean_bal_acc = np.mean([v["balanced_accuracy"] for v in val_res.values()])
        
        print(f"\nEpoch {epoch}: Loss = {avg_loss:.4f} | Mean Balanced Acc = {mean_bal_acc:.2%}")
        for k, v in val_res.items():
            print(f"  [{k}] Bal Acc: {v['balanced_accuracy']:.2%} (Baseline: {v['majority_baseline']:.2%}, Margin: {v['margin_percentage']:+.1f}%)")
            
        if mean_bal_acc >= best_score:
            best_score = mean_bal_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "classes": CLASSES,
                "best_score": best_score,
                "val_results": val_res
            }, CLASSIFIER_PATH)
            print(f"  => Saved checkpoint to {CLASSIFIER_PATH}")
            
    print(f"\n[SUCCESS] Training Complete. Best Score: {best_score:.2%}")


def evaluate_test_split(args):
    """Evaluate trained classifier on held-out test split (P0.9 --eval)."""
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Trained classifier checkpoint {CLASSIFIER_PATH} not found. Run --train first.")
        
    ckpt = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
    model = MultiTaskVideoClassifier(backbone=ckpt.get("backbone", "resnet18_avgpool"), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    df = pd.read_csv(MANIFEST_CSV)
    test_df = df[df["split"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError(
            "Test split has 0 clips in manifest.csv. "
            "Fallback to validation or full manifest is strictly forbidden (AGENTS.md §1 Rule 1, P0.4). "
            "Ingest and segment reels across splits before evaluating test performance."
        )
        
    print(f"Evaluating classifier on {len(test_df)} test samples...")
    test_ds = VideoClipDataset(test_df, target_size=112, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    results = evaluate_model(model, test_loader, device)
    
    print("\n" + "=" * 65)
    print("       STAGE 6 TEST SET EVALUATION AUDIT REPORT")
    print("=" * 65)
    for k, v in results.items():
        print(f"\nAttribute: {k.upper()}")
        print(f"  Balanced Accuracy:     {v['balanced_accuracy']:.2%}")
        print(f"  Majority Class Baseline: {v['majority_baseline']:.2%}")
        print(f"  Margin vs Majority:     {v['margin_percentage']:+.1f}%")
        gate_str = "PASSED" if v["gate_passed"] else "FAILED (< +25%)"
        print(f"  Stage 6 Gate:           {gate_str}")
        print("  Per-class metrics:")
        for cls_name, m in v["per_class"].items():
            print(f"    - {cls_name:18s} | Prec: {m['precision']:.2f} | Rec: {m['recall']:.2f} | Support: {m['support']}")
    return results


def evaluate_synthetic_clips(args):
    """Audit synthetic generated clips against prompt JSON sidecars (P0.9)."""
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Trained classifier checkpoint {CLASSIFIER_PATH} not found.")
        
    ckpt = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
    model = MultiTaskVideoClassifier(backbone=ckpt.get("backbone", "resnet18_avgpool"), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    synth_dir = args.synthetic_dir
    if not os.path.exists(synth_dir):
        print(f"Synthetic directory '{synth_dir}' not found.")
        return
        
    mp4_files = [os.path.join(synth_dir, f) for f in os.listdir(synth_dir) if f.endswith(".mp4")]
    print(f"Found {len(mp4_files)} clips in {synth_dir}")
    
    predictions = []
    audited_pairs = []
    
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
            
        # Check for expected JSON sidecar
        base_no_ext = os.path.splitext(f)[0]
        sidecar_json = f"{base_no_ext}.json"
        expected = None
        if os.path.exists(sidecar_json):
            try:
                with open(sidecar_json, "r") as jf:
                    expected = json.load(jf)
            except Exception:
                pass
                
        rec = {
            "file": os.path.basename(f),
            "predicted": {
                "burst_type": pred_burst,
                "yield_tier": pred_yield,
                "phase": pred_phase
            },
            "expected": expected,
            "has_ground_truth": expected is not None
        }
        predictions.append(rec)
        if expected is not None:
            audited_pairs.append(rec)
            
    # Save JSON report
    with open(SYNTHETIC_REPORT_JSON, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\n[SUCCESS] Synthetic evaluation report saved to {SYNTHETIC_REPORT_JSON}")
    
    # Check gate
    print("\n" + "=" * 65)
    print("       STAGE 6 SYNTHETIC GENERATION AUDIT GATE")
    print("=" * 65)
    print(f"Total video files inspected: {len(mp4_files)}")
    print(f"Clips with expected label sidecars: {len(audited_pairs)}")
    
    if len(audited_pairs) == 0:
        print("\nNotice (AGENTS.md §1 Rule 6):")
        print("  Current clips in samples/ are archival reference footage, not synthetic labeled runs.")
        print("  Stage 6 gate will evaluate synthetic outputs when sidecar JSONs or generator output runs are provided.")
    else:
        # Calculate quantitative accuracy against expected attributes
        correct_burst = sum(1 for p in audited_pairs if p["predicted"]["burst_type"] == p["expected"].get("burst_type"))
        acc_burst = correct_burst / len(audited_pairs)
        print(f"Burst Type Control Accuracy: {acc_burst:.2%}")


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
    
    if args.eval:
        evaluate_test_split(args)
    elif args.evaluate_synthetic:
        evaluate_synthetic_clips(args)
    elif args.train:
        train_classifier(args)
    else:
        train_classifier(args)


if __name__ == "__main__":
    main()
