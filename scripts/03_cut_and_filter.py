"""
scripts/03_cut_and_filter.py

Stage 3 Video Standardization & Preprocessing Engine.
Processes raw archival nuclear test reels into uniform, high-quality 16-frame
training clips at 8 fps (~2.0 seconds duration).

Fixes P0.5, P0.6, P1 items, and Non-Negotiable Rules:
- Matches clips to ground truth via data/shot_segments.csv and official splits;
  NEVER falls back to shot_df.iloc[0].
- Flash-onset detection and physical timing: determines phase by elapsed time
  since detonation:
    * early-fireball: 0.0 <= delta_t <= 0.5s
    * expansion:      0.5 <  delta_t <= 2.5s
    * stabilized-cloud: delta_t > 2.5s
- Caps clips per (shot, phase) to prevent stabilized-cloud dominance
- Stores detonation_start_sec and offset_sec in manifest.csv
- Uses canonical schema.py (burst_type, yield_tier, terrain, lighting, phase, film_look, smoke_trails)
- Strict POSIX '/' paths written to manifest.csv
- Rebuild mode (--rebuild) to regenerate manifest cleanly without stale rows
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import schema

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "raw_videos")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_clips")
SPLITS_CSV = os.path.join(DATA_DIR, "doe_nv209_atmospheric_splits.csv")
SEGMENTS_CSV = os.path.join(DATA_DIR, "shot_segments.csv")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")

TARGET_FPS = 8
CLIP_FRAMES = 16  # Exactly 2.0 seconds at 8 fps
MAX_CLIPS_PER_PHASE = 12  # Balance phase distribution


def load_shot_metadata():
    """Load stratified shot metadata."""
    if not os.path.exists(SPLITS_CSV):
        raise FileNotFoundError(f"{SPLITS_CSV} not found! Run scripts/02b_audit_class_counts.py first.")
    return pd.read_csv(SPLITS_CSV)


def load_segments_metadata():
    """Load shot segments metadata if present."""
    if os.path.exists(SEGMENTS_CSV):
        return pd.read_csv(SEGMENTS_CSV)
    return pd.DataFrame()


def detect_scenes(video_path, threshold=27.0):
    """Run PySceneDetect to find camera cut boundaries."""
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=15))
        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()
        if not scenes:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return [(0.0, total_frames / fps)]
        return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]
    except Exception as e:
        print(f"Warning: Scene detection failed for {video_path}: {e}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return [(0.0, total_frames / fps)]


def find_detonation_flash_sec(video_path, max_search_sec=90.0):
    """Detect detonation flash timestamp from sudden dramatic luminance surge."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps / 4.0))  # Sample 4 times a second
    
    times = []
    lums = []
    f_idx = 0
    max_frames = int(min(total_frames, max_search_sec * fps))
    
    while f_idx < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lums.append(float(np.mean(gray)))
        times.append(f_idx / fps)
        f_idx += step
    cap.release()
    
    if len(lums) < 5:
        return 0.0
        
    diffs = np.diff(lums)
    max_diff_idx = int(np.argmax(diffs))
    if diffs[max_diff_idx] > 20.0:
        return float(times[max_diff_idx])
    return 0.0


def crop_and_resize(frame, target_size=512):
    """Center crop frame to square and resize to target_size x target_size."""
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    cropped = frame[top:top + min_dim, left:left + min_dim]
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)


def match_source_to_shot(video_filename, shot_df, segments_df):
    """
    Match video file to official shot record without fallback to iloc[0].
    Enforces segment confidence >= 0.8 and strictly rejects compilation reels
    lacking verified shot boundaries in shot_segments.csv.
    """
    fname_clean = os.path.basename(video_filename)
    
    # 1. Match via segments_df with confidence check
    if not segments_df.empty:
        for _, s_row in segments_df.iterrows():
            aid = str(s_row.get("archive_id", "")).lower()
            conf = float(s_row.get("confidence", 0.0))
            if conf >= 0.8 and aid and aid in fname_clean.lower():
                op = s_row["operation"]
                shot = s_row["shot_name"]
                m = shot_df[(shot_df["operation"] == op) & (shot_df["shot_name"] == shot)]
                if not m.empty:
                    return m.iloc[0], float(s_row.get("detonation_start_sec", 0.0))

    # Reject compilation reels lacking explicit verified segments
    compilation_markers = ["compilation", "tests", "nuclear test film", "operation plumbbob", "operation hardtack", "composite"]
    if any(marker in fname_clean.lower() for marker in compilation_markers):
        # Unless a specific shot name is uniquely present in the filename
        matched_shots = []
        for _, row in shot_df.iterrows():
            s_name = str(row["shot_name"]).lower()
            if len(s_name) > 3 and s_name in fname_clean.lower():
                matched_shots.append(row)
        if len(matched_shots) != 1:
            print(f"Rejecting compilation reel '{fname_clean}': lacks verified shot segments in shot_segments.csv.")
            return None, None
        return matched_shots[0], None

    # 2. Match directly by shot name
    for _, row in shot_df.iterrows():
        s_name = str(row["shot_name"]).lower()
        if len(s_name) > 3 and s_name in fname_clean.lower():
            return row, None
            
    # 3. Match by operation + unique match
    candidates = []
    for _, row in shot_df.iterrows():
        op_name = str(row["operation"]).lower()
        if len(op_name) > 3 and op_name in fname_clean.lower():
            candidates.append(row)
    if len(candidates) == 1:
        return candidates[0], None
        
    # Strictly fail on unknown source: no silent fallback to shot_df.iloc[0]!
    return None, None


def process_video_file(video_path, output_dir, shot_df, segments_df, resolution=512, augment=True):
    """Process an archival video file into standardized 16-frame clips."""
    shot_match, detonation_sec_known = match_source_to_shot(video_path, shot_df, segments_df)
    if shot_match is None:
        print(f"Skipping {video_path}: No matching historical shot in database.")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_basename = os.path.basename(video_path)

    if total_frames < 30:
        cap.release()
        return []

    detonation_start_sec = detonation_sec_known
    if detonation_start_sec is None or detonation_start_sec <= 0.0:
        detonation_start_sec = find_detonation_flash_sec(video_path)

    operation = shot_match["operation"]
    shot_name = shot_match["shot_name"]
    burst_type = shot_match["burst_type"]
    yield_tier = shot_match["yield_tier"]
    terrain = shot_match["terrain"]
    lighting = shot_match["lighting"]
    split = shot_match.get("split", "train")

    scenes = detect_scenes(video_path)
    frame_step = max(1, int(round(src_fps / TARGET_FPS)))

    clip_records = []
    phase_counts = {"early-fireball": 0, "expansion": 0, "stabilized-cloud": 0}
    clip_counter = 0

    for scene_idx, (s_start, s_end) in enumerate(scenes):
        scene_dur = s_end - s_start
        if scene_dur < 2.0:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s_start * src_fps))
        raw_frames = []
        for _ in range(int(scene_dur * src_fps)):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            raw_frames.append(frame)

        if len(raw_frames) < frame_step * CLIP_FRAMES:
            continue

        subsampled = raw_frames[::frame_step]
        window_idx = 0

        while window_idx + CLIP_FRAMES <= len(subsampled):
            window_frames = subsampled[window_idx:window_idx + CLIP_FRAMES]
            clip_time_sec = s_start + (window_idx * frame_step) / src_fps
            offset_sec = clip_time_sec - detonation_start_sec

            # Skip pre-detonation countdown footage
            if detonation_start_sec > 0.0 and offset_sec < -0.5:
                window_idx += CLIP_FRAMES // 2
                continue

            # Assign physical detonation phase by elapsed time (AGENTS.md P0.6)
            if offset_sec <= 0.5:
                phase = "early-fireball"
            elif offset_sec <= 2.5:
                phase = "expansion"
            else:
                phase = "stabilized-cloud"

            # Check phase cap to ensure class balance
            if phase_counts[phase] >= MAX_CLIPS_PER_PHASE:
                window_idx += CLIP_FRAMES // 2
                continue

            processed_frames = []
            is_color_votes = []
            motion_deltas = []

            for f in window_frames:
                proc = crop_and_resize(f, resolution)
                processed_frames.append(proc)
                b, g, r = cv2.split(proc)
                channel_var = (np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32))) +
                               np.mean(np.abs(r.astype(np.float32) - b.astype(np.float32)))) / 2.0
                is_color_votes.append(channel_var > 7.0)

            for i in range(len(processed_frames) - 1):
                g1 = cv2.cvtColor(processed_frames[i], cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(processed_frames[i+1], cv2.COLOR_BGR2GRAY)
                motion_deltas.append(float(np.mean(cv2.absdiff(g1, g2))))

            avg_motion = float(np.mean(motion_deltas))
            if avg_motion < 0.6:  # Skip static frames
                window_idx += CLIP_FRAMES // 2
                continue

            film_look = "colour-restored" if (sum(is_color_votes) / len(is_color_votes)) > 0.6 else "black-and-white"

            # Smoke trails detection (vertical diagnostic lines in initial frame)
            edges = cv2.Canny(cv2.cvtColor(processed_frames[0], cv2.COLOR_BGR2GRAY), 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)
            has_smoke_trails = False
            if lines is not None:
                vert_count = sum(
                    1 for l in lines if len(l.flatten()) >= 4 and
                    75.0 <= np.abs(np.arctan2(l.flatten()[3] - l.flatten()[1], l.flatten()[2] - l.flatten()[0]) * 180.0 / np.pi) <= 105.0
                )
                has_smoke_trails = vert_count >= 3

            clip_counter += 1
            phase_counts[phase] += 1
            
            # Unique clip prefix incorporating source video identity
            source_id = os.path.splitext(video_basename)[0][:20].replace(" ", "_")
            base_clip_id = f"{operation}_{shot_name}_{source_id}_c{clip_counter:04d}"

            variants = [("orig", processed_frames)]
            if augment:
                flipped = [cv2.flip(f, 1) for f in processed_frames]
                variants.append(("flip", flipped))

            for var_name, var_frames in variants:
                clip_id = f"{base_clip_id}_{var_name}"
                out_filename = f"{clip_id}.mp4"
                out_path = os.path.join(output_dir, out_filename)
                rel_clip_path = f"processed_clips/{out_filename}"  # Strict POSIX '/'

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, TARGET_FPS, (resolution, resolution))
                for frame in var_frames:
                    writer.write(frame)
                writer.release()

                # Format standardized prompt using canonical schema
                prompt = schema.format_prompt(
                    burst_type=burst_type,
                    yield_tier=yield_tier,
                    terrain=terrain,
                    lighting=lighting,
                    phase=phase,
                    film_look=film_look,
                    smoke_trails=has_smoke_trails
                )
                schema.validate_prompt(prompt)

                clip_records.append({
                    "clip_id": clip_id,
                    "operation": operation,
                    "shot_name": shot_name,
                    "split": split,
                    "burst_type": burst_type,
                    "yield_tier": yield_tier,
                    "terrain": terrain,
                    "lighting": lighting,
                    "setting": f"{terrain}-{lighting}" if terrain == "desert-flats" else terrain,
                    "phase": phase,
                    "film_look": film_look,
                    "smoke_trails": has_smoke_trails,
                    "prompt": prompt,
                    "clip_path": rel_clip_path,
                    "latent_path": None,
                    "start_sec": round(clip_time_sec, 2),
                    "duration_sec": 2.0,
                    "detonation_start_sec": round(detonation_start_sec, 2),
                    "offset_sec": round(offset_sec, 2),
                    "resolution": resolution,
                    "fps": TARGET_FPS,
                    "variant": var_name,
                    "source_reel": video_basename
                })

            window_idx += CLIP_FRAMES // 2

    cap.release()
    print(f"Extracted {len(clip_records)} clips from {video_basename} (Phases: {phase_counts})")
    return clip_records


def run_preprocessing(input_dir, output_dir, resolution=512, single_file=None, rebuild=False):
    """Run preprocessing across all archival video reels."""
    os.makedirs(output_dir, exist_ok=True)
    shot_df = load_shot_metadata()
    segments_df = load_segments_metadata()

    if single_file:
        video_files = [single_file]
    else:
        video_files = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        ]

    print(f"Found {len(video_files)} video reels to process.")
    all_clips = []
    
    for v_path in video_files:
        clips = process_video_file(v_path, output_dir, shot_df, segments_df, resolution=resolution)
        all_clips.extend(clips)

    if not all_clips:
        print("No clips extracted.")
        return

    new_df = pd.DataFrame(all_clips)

    if os.path.exists(MANIFEST_CSV) and not rebuild:
        old_df = pd.read_csv(MANIFEST_CSV)
        # Avoid duplicate clips
        merged_df = pd.concat([old_df[~old_df["clip_id"].isin(new_df["clip_id"])], new_df], ignore_index=True)
    else:
        merged_df = new_df

    # Ensure POSIX '/' in clip paths and latent paths
    merged_df["clip_path"] = merged_df["clip_path"].apply(lambda p: str(p).replace("\\", "/") if pd.notna(p) else p)
    if "latent_path" in merged_df.columns:
        merged_df["latent_path"] = merged_df["latent_path"].apply(lambda p: str(p).replace("\\", "/") if pd.notna(p) else p)

    merged_df.to_csv(MANIFEST_CSV, index=False)
    print(f"\n[SUCCESS] Manifest updated at {MANIFEST_CSV}: {len(merged_df)} total clips.")
    print("Phase distribution:")
    print(merged_df["phase"].value_counts().to_string())
    print("\nSplit distribution:")
    print(merged_df["split"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Video Standardization & Preprocessing")
    parser.add_argument("--single-file", type=str, default=None, help="Process single video file")
    parser.add_argument("--input-dir", type=str, default="raw_videos", help="Input directory")
    parser.add_argument("--output-dir", type=str, default="processed_clips", help="Output directory")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution (512)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild manifest cleanly")
    args = parser.parse_args()

    run_preprocessing(
        args.input_dir,
        args.output_dir,
        resolution=args.resolution,
        single_file=args.single_file,
        rebuild=args.rebuild
    )
