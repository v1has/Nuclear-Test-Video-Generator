#!/usr/bin/env python3
"""
scripts/02c_segment_reels.py

Stage 2c: Automated Archival Reel Segmentation & Shot Identification Engine.
Solves Step 2 of the critical path (AGENTS.md §3.1, §3.3):
1. Detects scene cut boundaries using PySceneDetect (with OpenCV fallback).
2. OCRs title cards, slates, and introductory text overlays using EasyOCR / PyTesseract.
3. Fuzzy-matches OCR text to the official 211-shot DOE/NV-209 registry using RapidFuzz.
4. Detects detonation flash onsets to compute exact detonation_start_sec.
5. Emits candidate segments with confidence scores and provenance.
6. Merges reviewed high-confidence rows into data/shot_segments.csv.

Usage:
  python scripts/02c_segment_reels.py --archive_id gov.doe.0800002
  python scripts/02c_segment_reels.py --all
  python scripts/02c_segment_reels.py --all --apply --min_confidence 0.8
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm

# Fuzzy string matching
try:
    from rapidfuzz import fuzz
except ImportError:
    from difflib import SequenceMatcher
    class FuzzFallback:
        @staticmethod
        def token_set_ratio(s1, s2):
            return SequenceMatcher(None, s1.lower(), s2.lower()).ratio() * 100.0
        @staticmethod
        def partial_ratio(s1, s2):
            return SequenceMatcher(None, s1.lower(), s2.lower()).ratio() * 100.0
    fuzz = FuzzFallback()

# PySceneDetect
try:
    from scenedetect import open_video, SceneManager, ContentDetector
    HAS_SCENEDETECT = True
except ImportError:
    HAS_SCENEDETECT = False

# OCR Engine (EasyOCR preferred, fallback to pytesseract)
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
RAW_VIDEOS_DIR = REPO_ROOT / "raw_videos"
SOURCES_CSV = DATA_DIR / "sources.csv"
DOE_SHOTS_CSV = DATA_DIR / "doe_nv209_atmospheric_shots.csv"
SEGMENTS_CSV = DATA_DIR / "shot_segments.csv"
CANDIDATES_CSV = DATA_DIR / "shot_segments_candidates.csv"

_EASYOCR_READER = None


def get_ocr_reader():
    """Lazy initialize EasyOCR Reader."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None and HAS_EASYOCR:
        use_gpu = torch.cuda.is_available()
        print(f"Initializing EasyOCR reader (gpu={use_gpu})...", flush=True)
        try:
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        except Exception as e:
            print(f"Warning: Failed to initialize EasyOCR on GPU={use_gpu} ({e}). Retrying on CPU...", flush=True)
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _EASYOCR_READER


def extract_ocr_text(frame_rgb):
    """Extract recognized text from RGB video frame using available OCR engines."""
    text_results = []
    
    # 1. Try EasyOCR
    reader = get_ocr_reader()
    if reader is not None:
        try:
            res = reader.readtext(frame_rgb, detail=0)
            if res:
                text_results.extend(res)
        except Exception:
            pass

    # 2. Try PyTesseract fallback if EasyOCR yielded nothing or not available
    if not text_results and HAS_TESSERACT:
        try:
            raw_text = pytesseract.image_to_string(frame_rgb)
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            text_results.extend(lines)
        except Exception:
            pass

    clean_text = " ".join(text_results).strip()
    return clean_text


def detect_scene_cuts(video_path, threshold=27.0, min_scene_len_sec=1.5):
    """Detect cut boundaries using PySceneDetect with OpenCV fallback."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()

    if duration_sec <= 0:
        return []

    print(f"Scanning scene cuts in {video_path.name} ({duration_sec:.1f}s, {total_frames} frames)...", flush=True)
    scenes = []
    if HAS_SCENEDETECT:
        try:
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=int(fps * min_scene_len_sec)))
            # frame_skip=3 evaluates 1 frame out of 3, speeding up 3-4x while maintaining cut accuracy
            scene_manager.detect_scenes(video, frame_skip=3)
            detected = scene_manager.get_scene_list()
            for start_tc, end_tc in detected:
                scenes.append((float(start_tc.seconds), float(end_tc.seconds)))
        except Exception as e:
            print(f"PySceneDetect error ({e}), falling back to adaptive OpenCV detector.", flush=True)
            scenes = []

    # OpenCV fallback if scenedetect produced nothing or failed
    if not scenes:
        cap = cv2.VideoCapture(str(video_path))
        prev_hist = None
        scene_start = 0.0
        frame_idx = 0
        step = max(1, int(fps / 4))  # sample 4 times per second for speed

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_idx % step == 0:
                current_time = frame_idx / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                if prev_hist is not None:
                    corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    if corr < 0.65 and (current_time - scene_start) >= min_scene_len_sec:
                        scenes.append((scene_start, current_time))
                        scene_start = current_time
                prev_hist = hist
            frame_idx += 1

        cap.release()
        if duration_sec - scene_start >= min_scene_len_sec:
            scenes.append((scene_start, duration_sec))

    if not scenes:
        scenes = [(0.0, duration_sec)]
    return scenes


def find_detonation_flash_sustained(video_path, start_sec, end_sec, min_sustained_sec=1.0):
    """
    Detect detonation flash onset characterized by:
    1. A sharp luminance rise (jump >= 20.0).
    2. A sustained bright plateau over at least min_sustained_sec (e.g. 1.0 - 1.5s),
       distinguishing genuine nuclear fireballs from momentary scene cuts or bright sky.
    Returns (onset_sec, boundary_confidence).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return start_sec, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Sample every 2-3 frames for speed and precision
    step = max(1, int(fps / 15.0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    lums = []
    times = []

    curr_frame = start_frame
    while curr_frame <= end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if (curr_frame - start_frame) % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lums.append(float(np.mean(gray)))
            times.append(curr_frame / fps)
        curr_frame += 1

    cap.release()
    if len(lums) < 5:
        return start_sec, 0.0

    lums = np.array(lums)
    times = np.array(times)
    diffs = np.diff(lums)

    best_onset = start_sec
    best_conf = 0.0
    spike_indices = np.where(diffs >= 20.0)[0]
    sustained_samples = max(2, int(min_sustained_sec * (fps / step)))

    for s_idx in spike_indices:
        t_onset = times[s_idx + 1]
        pre_lum = lums[max(0, s_idx - 2):s_idx + 1].mean()
        post_lums = lums[s_idx + 1:min(len(lums), s_idx + 1 + sustained_samples)]

        if len(post_lums) < 2:
            continue

        mean_post = float(np.mean(post_lums))
        min_post = float(np.min(post_lums))

        # Sustained check: post brightness is high, elevated over pre, and does not drop immediately
        if mean_post >= 60.0 and (mean_post - pre_lum) >= 25.0 and min_post >= (pre_lum + 10.0):
            conf = min(1.0, float((mean_post - pre_lum) / 70.0) * 0.7 + (float(diffs[s_idx]) / 100.0) * 0.3)
            if conf > best_conf:
                best_conf = conf
                best_onset = float(t_onset)

    return best_onset, best_conf


def match_text_to_shot(text, operation_hint, registry_df):
    """
    Match OCR text against candidate shots in DOE registry.
    Supports compound operation hints (e.g. 'Ranger / Buster-Jangle').
    Returns (matched_row, confidence, match_type).
    """
    if not text or len(text.strip()) < 2:
        return None, 0.0, "none"

    clean = re.sub(r"[^A-Za-z0-9\s\-]", " ", text).strip()

    # Filter candidate shots: prefer operation_hint if provided
    candidates = registry_df
    sub_hints = []
    if operation_hint and operation_hint.lower() != "compilation":
        sub_hints = [h.strip().lower() for h in re.split(r"[/,]", operation_hint) if h.strip()]
        op_candidates = registry_df[
            registry_df["operation"].str.lower().apply(lambda op: any(h in op or op in h for h in sub_hints))
        ]
        if not op_candidates.empty:
            candidates = op_candidates

    best_match = None
    best_score = 0.0
    best_type = "none"

    for _, row in candidates.iterrows():
        shot_name = str(row["shot_name"]).strip()
        shot_lower = shot_name.lower()
        op_name = str(row["operation"]).strip().lower()

        # 1. Exact token search: "shot baker", "baker", "test able", "charlie"
        pattern = rf"\b{re.escape(shot_lower)}\b"
        if re.search(pattern, clean.lower()):
            score = 0.95
            if any(h in clean.lower() for h in [op_name] + sub_hints):
                score = 1.0
            if score > best_score:
                best_score = score
                best_match = row
                best_type = "exact_token"
            continue

        # 2. Fuzzy ratio using token_set_ratio
        ratio = fuzz.token_set_ratio(shot_lower, clean.lower()) / 100.0
        if ratio >= 0.80 and ratio > best_score:
            best_score = ratio * 0.90
            best_match = row
            best_type = "fuzzy_ratio"

    return best_match, best_score, best_type


def segment_video_reel(video_path, archive_id, operation_hint, registry_df, min_confidence=0.65):
    """
    Segment a single archival reel:
    - Finds scene cuts
    - Samples frames for OCR
    - Identifies shot name from slates/title cards
    - Detects detonation flash using sustained-luminance analysis
    """
    print(f"\n--- Segmenting Reel: {video_path.name} (Op: {operation_hint}, ID: {archive_id}) ---", flush=True)
    scenes = detect_scene_cuts(video_path)
    print(f"Detected {len(scenes)} distinct scene cuts.", flush=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    detected_segments = []
    
    for i, (s_start, s_end) in enumerate(scenes):
        scene_dur = s_end - s_start
        if scene_dur < 1.0:
            continue

        # Sample keyframe from this scene to OCR (at 0.5s or midpoint)
        sample_time = s_start + min(1.0, scene_dur * 0.5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sample_time * fps))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        combined_text = extract_ocr_text(rgb)

        if not combined_text:
            continue

        matched_shot, ocr_conf, mtype = match_text_to_shot(combined_text, operation_hint, registry_df)
        if matched_shot is not None and ocr_conf >= min_confidence:
            shot_name = matched_shot["shot_name"]
            shot_op = matched_shot["operation"]
            burst_type = matched_shot.get("burst_type", "")
            print(f"  [Scene {i:02d} @ {s_start:6.1f}s - {s_end:6.1f}s] Matched '{shot_op} / {shot_name}' (OCR conf: {ocr_conf:.2f}, text: '{combined_text[:60]}...')", flush=True)

            # Search forward through reel until next slate or end
            search_window_end = scenes[-1][1]
            if i + 1 < len(scenes):
                # Search up to next 500s or end
                search_window_end = min(s_start + 600.0, scenes[-1][1])

            flash_sec, bnd_conf = find_detonation_flash_sustained(video_path, s_start, search_window_end)
            if bnd_conf > 0.0:
                print(f"    -> Sustained flash detected at {flash_sec:.2f}s (offset: {flash_sec - s_start:.2f}s from slate, bnd_conf: {bnd_conf:.2f})", flush=True)
                seg_end = min(flash_sec + 120.0, scenes[-1][1])
                seg_start = max(0.0, flash_sec - 5.0)
            else:
                flash_sec = s_start
                seg_end = min(s_end + 90.0, scenes[-1][1])
                seg_start = s_start

            combined_conf = round(ocr_conf * 0.6 + bnd_conf * 0.4, 3)

            detected_segments.append({
                "archive_id": archive_id,
                "operation": shot_op,
                "shot_name": shot_name,
                "start_sec": round(seg_start, 2),
                "end_sec": round(seg_end, 2),
                "detonation_start_sec": round(flash_sec, 2),
                "confidence": combined_conf,
                "ocr_confidence": round(ocr_conf, 3),
                "boundary_confidence": round(bnd_conf, 3),
                "review_status": "draft",
                "label_source": f"PySceneDetect+OCR_{mtype}+SustainedFlash",
                "reviewer": "auto_segmenter",
                "film_look_override": None,
                "smoke_trails_override": False if burst_type == "underwater" else None
            })

    cap.release()
    return detected_segments

    cap.release()
    return detected_segments


def load_authoritative_metadata():
    """Load DOE NV-209 shots registry and sources manifest."""
    if not DOE_SHOTS_CSV.exists():
        raise FileNotFoundError(f"Authoritative registry not found: {DOE_SHOTS_CSV}")
    registry_df = pd.read_csv(DOE_SHOTS_CSV)
    
    sources_df = None
    if SOURCES_CSV.exists():
        sources_df = pd.read_csv(SOURCES_CSV)
    return registry_df, sources_df


def run_segmentation(archive_ids=None, video_file=None, min_confidence=0.75, apply_to_manifest=False):
    """Run reel segmentation pipeline."""
    registry_df, sources_df = load_authoritative_metadata()
    targets = []

    if video_file:
        vpath = Path(video_file)
        if not vpath.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
        targets.append({
            "path": vpath,
            "archive_id": vpath.stem,
            "operation": "Compilation"
        })
    elif sources_df is not None:
        if archive_ids:
            if isinstance(archive_ids, str):
                id_list = [i.strip() for i in archive_ids.split(",") if i.strip()]
            else:
                id_list = list(archive_ids)
            filtered = sources_df[sources_df["archive_id"].isin(id_list)]
        else:
            filtered = sources_df

        for _, row in filtered.iterrows():
            aid = row["archive_id"]
            op = row.get("operation", "Compilation")
            lpath = REPO_ROOT / row["local_path"]
            if lpath.exists() and lpath.stat().st_size > 1024:
                targets.append({
                    "path": lpath,
                    "archive_id": aid,
                    "operation": op
                })
            else:
                print(f"Skipping {aid}: Video not downloaded yet ({lpath})")

    if not targets:
        print("No downloaded video files found to segment. Run scripts/01_download_archive.py --download first.")
        return []

    print(f"Commencing segmentation across {len(targets)} video reel(s)...")
    all_candidates = []

    for t in targets:
        segs = segment_video_reel(
            video_path=t["path"],
            archive_id=t["archive_id"],
            operation_hint=t["operation"],
            registry_df=registry_df,
            min_confidence=min_confidence
        )
        all_candidates.extend(segs)

    if not all_candidates:
        print("\nNo candidate shot segments detected with current confidence threshold.")
        return []

    candidates_df = pd.DataFrame(all_candidates)
    candidates_df.to_csv(CANDIDATES_CSV, index=False)
    print(f"\n[OUTPUT] Saved {len(candidates_df)} candidate segment(s) to {CANDIDATES_CSV}")

    # Merge into data/shot_segments.csv if requested (preserving all verified entries)
    if apply_to_manifest:
        high_conf = candidates_df[candidates_df["confidence"] >= min_confidence]
        if not high_conf.empty:
            if SEGMENTS_CSV.exists():
                existing = pd.read_csv(SEGMENTS_CSV)
                # Preserve all verified / reviewed entries unconditionally
                verified_mask = existing["review_status"].isin(["verified", "reviewed"]) if "review_status" in existing.columns else pd.Series([False]*len(existing))
                verified_keys = set(zip(
                    existing[verified_mask]["archive_id"],
                    existing[verified_mask]["operation"],
                    existing[verified_mask]["shot_name"]
                ))
                
                new_rows = []
                for _, r in high_conf.iterrows():
                    key = (r["archive_id"], r["operation"], r["shot_name"])
                    if key not in verified_keys:
                        new_rows.append(r)
                if new_rows:
                    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["archive_id", "operation", "shot_name"], keep="first")
                else:
                    combined = existing
                    print("  Note: All candidate shots already have verified entries; preserved verified rows.")
            else:
                combined = high_conf
            combined.to_csv(SEGMENTS_CSV, index=False)
            print(f"[APPLIED] Maintained verified segments in {SEGMENTS_CSV}. Total: {len(combined)}")

    return all_candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2c: Archival Reel Segmentation & Shot Identification Engine")
    parser.add_argument("--archive_id", type=str, default=None, help="Archive ID from data/sources.csv to segment")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated archive IDs to segment")
    parser.add_argument("--video", type=str, default=None, help="Direct path to video file to segment")
    parser.add_argument("--all", action="store_true", help="Segment all downloaded video reels")
    parser.add_argument("--min_confidence", type=float, default=0.75, help="Minimum confidence threshold (0.0 - 1.0)")
    parser.add_argument("--apply", action="store_true", help="Merge high-confidence segments into data/shot_segments.csv")
    args = parser.parse_args()

    ids_to_process = args.ids or args.archive_id
    run_segmentation(
        archive_ids=ids_to_process,
        video_file=args.video,
        min_confidence=args.min_confidence,
        apply_to_manifest=args.apply
    )
