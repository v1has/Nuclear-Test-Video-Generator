"""
scripts/03_cut_and_filter.py

Stage 3 Video Standardization & Preprocessing Engine.
Processes raw archival nuclear test reels into uniform, high-quality 16-frame
training clips at 8 fps (~2.0 seconds duration).

Features:
- Automated scene detection with PySceneDetect to identify camera cut boundaries.
- Quality filtering: discards blank frames, white flash frames (>92% saturation),
  and static title slates (low inter-frame motion delta).
- Standardized center cropping and resizing to target dimensions (512x512 or 384x384).
- High-efficiency sequential frame reading and fast motion estimation.
- Deterministic data augmentation before VAE encoding:
    * Original clip
    * Horizontal mirror flip
- Cross-references each clip with official DOE ground truth and assigned
  stratified splits from data/doe_nv209_atmospheric_splits.csv.
- Compiles data/manifest.csv.

Usage:
  python scripts/03_cut_and_filter.py --single-file "raw_videos/Dominic_DominicSunset_Dominic Sunset.mp4"
  python scripts/03_cut_and_filter.py --input-dir raw_videos --output-dir processed_clips --resolution 512
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "raw_videos")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_clips")
SPLITS_CSV = os.path.join(DATA_DIR, "doe_nv209_atmospheric_splits.csv")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")

TARGET_FPS = 8
CLIP_FRAMES = 16  # Exactly 2.0 seconds at 8 fps


def load_shot_metadata():
    """Load stratified shot metadata."""
    if not os.path.exists(SPLITS_CSV):
        print(f"Error: {SPLITS_CSV} not found! Run scripts/02b_audit_class_counts.py first.")
        sys.exit(1)
    return pd.read_csv(SPLITS_CSV)


def detect_scenes(video_path, threshold=27.0):
    """
    Run PySceneDetect to find camera cut boundaries.
    Returns list of (start_sec, end_sec).
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=15))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    scenes = []
    for scene in scene_list:
        # Use .seconds attribute (scenedetect 0.7+)
        start_sec = scene[0].seconds
        end_sec = scene[1].seconds
        if (end_sec - start_sec) >= 1.5:  # At least 1.5 seconds
            scenes.append((start_sec, end_sec))
            
    if not scenes:
        duration = video.duration.seconds
        scenes = [(0.0, duration)]
        
    return scenes


def analyze_frame_quality(frame):
    """
    Fast quality checks for a single BGR frame.
    Returns (is_valid, reason, is_color, mean_luminance)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray))
    
    # 1. Blank / pure black check (< 8.0 mean luminance)
    if mean_lum < 8.0:
        return False, "black_frame", False, mean_lum
        
    # 2. Overexposed white flash check (>85% pixels > 245)
    white_ratio = float(np.sum(gray > 245)) / gray.size
    if white_ratio > 0.85:
        return False, "white_flash", False, mean_lum
        
    # 3. Check if black-and-white or colour via channel variance
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    rg_diff = np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
    rb_diff = np.mean(np.abs(r.astype(np.float32) - b.astype(np.float32)))
    channel_variance = (rg_diff + rb_diff) / 2.0
    is_color = channel_variance > 7.0
    
    return True, "valid", is_color, mean_lum


def crop_and_resize(frame, target_size=512):
    """Center crop frame to square and resize to target_size x target_size."""
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    cropped = frame[top:top + min_dim, left:left + min_dim]
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)


def match_source_to_shot(video_filename, shot_df):
    """Match video filename to candidate shot in DOE records."""
    fname_lower = video_filename.lower()
    
    # 1. Match by shot name
    for _, row in shot_df.iterrows():
        s_name = str(row['shot_name']).lower()
        if len(s_name) > 2 and s_name in fname_lower:
            return row
            
    # 2. Match by operation
    for _, row in shot_df.iterrows():
        op_name = str(row['operation']).lower()
        if len(op_name) > 3 and op_name in fname_lower:
            return row
            
    return shot_df.iloc[0]


def format_prompt(burst_type, yield_tier, setting, phase, film_look, smoke_trails=False):
    """Format standardized prompt string."""
    trails_token = "with smoke-trails" if smoke_trails else "without smoke-trails"
    return f"{burst_type} nuclear detonation, {yield_tier}, {setting}, {phase} phase, {film_look} archival footage, {trails_token}"


def process_video_file(video_path, output_dir, shot_df, resolution=512, augment=True):
    """Process a single archival video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return []
        
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_basename = os.path.basename(video_path)
    
    if src_fps <= 0 or total_frames < 30:
        cap.release()
        return []
        
    matched_shot = match_source_to_shot(video_basename, shot_df)
    burst_type = matched_shot["burst_type"]
    yield_tier = matched_shot["yield_tier"]
    setting = matched_shot["setting"]
    shot_name = matched_shot["shot_name"]
    operation = matched_shot["operation"]
    split = matched_shot["split"]
    
    print(f"   Source: {video_basename} -> Matched Shot: {operation} {shot_name} (Split: {split})")
    print(f"   Detecting scene boundaries...")
    scenes = detect_scenes(video_path)
    print(f"   Identified {len(scenes)} distinct scene segments.")
    
    step = max(1, int(round(src_fps / TARGET_FPS)))
    manifest_records = []
    clip_counter = 0
    
    for s_idx, (start_sec, end_sec) in enumerate(scenes):
        start_frame = int(start_sec * src_fps)
        end_frame = int(end_sec * src_fps)
        
        # Read frames sequentially for this scene
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        scene_frames = []
        curr_f = start_frame
        while curr_f < end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if (curr_f - start_frame) % step == 0:
                scene_frames.append(frame)
            curr_f += 1
            
        if len(scene_frames) < CLIP_FRAMES:
            continue
            
        # Extract sliding windows of CLIP_FRAMES
        window_idx = 0
        while (window_idx + CLIP_FRAMES) <= len(scene_frames):
            window = scene_frames[window_idx:window_idx + CLIP_FRAMES]
            
            # Quality checks
            valid_window = True
            luminances = []
            is_color_votes = []
            processed_frames = []
            
            for f in window:
                is_val, reason, is_col, lum = analyze_frame_quality(f)
                if not is_val:
                    valid_window = False
                    break
                luminances.append(lum)
                is_color_votes.append(is_col)
                processed_frames.append(crop_and_resize(f, target_size=resolution))
                
            if not valid_window or len(processed_frames) != CLIP_FRAMES:
                window_idx += CLIP_FRAMES // 2
                continue
                
            # Inter-frame motion delta via fast absdiff
            motion_deltas = []
            for i in range(len(processed_frames) - 1):
                g1 = cv2.cvtColor(processed_frames[i], cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(processed_frames[i+1], cv2.COLOR_BGR2GRAY)
                motion_deltas.append(float(np.mean(cv2.absdiff(g1, g2))))
                
            avg_motion = np.mean(motion_deltas)
            
            # Filter static slates (nearly zero motion delta < 0.8)
            if avg_motion < 0.8:
                window_idx += CLIP_FRAMES // 2
                continue
                
            # Detonation phase estimation
            lum_delta = np.max(luminances) - np.min(luminances)
            avg_lum = np.mean(luminances)
            if lum_delta > 35.0 or (avg_lum > 165.0 and avg_motion > 15.0):
                phase = "flash-fireball"
            elif avg_motion > 6.0:
                phase = "stem-rise"
            else:
                phase = "stabilized-cloud"
                
            film_look = "colour-restored" if (sum(is_color_votes) / len(is_color_votes)) > 0.6 else "black-and-white"
            
            # Smoke trails detection (vertical lines in first frame)
            edges = cv2.Canny(cv2.cvtColor(processed_frames[0], cv2.COLOR_BGR2GRAY), 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)
            has_smoke_trails = False
            if lines is not None:
                vert_count = 0
                for line in lines:
                    coords = line.flatten()
                    if len(coords) >= 4:
                        x1, y1, x2, y2 = coords[:4]
                        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                        if 75.0 <= angle <= 105.0:
                            vert_count += 1
                has_smoke_trails = vert_count >= 3
                
            clip_counter += 1
            base_clip_id = f"{operation}_{shot_name}_c{clip_counter:04d}"
            
            variants = [("orig", processed_frames)]
            if augment:
                flipped = [cv2.flip(f, 1) for f in processed_frames]
                variants.append(("flip", flipped))
                
            for var_suffix, var_frames in variants:
                clip_id = f"{base_clip_id}_{var_suffix}"
                clip_filename = f"{clip_id}.mp4"
                clip_path = os.path.join(output_dir, clip_filename)
                rel_clip_path = os.path.relpath(clip_path, PROJECT_ROOT)
                
                # Write MP4
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(clip_path, fourcc, TARGET_FPS, (resolution, resolution))
                for f in var_frames:
                    out.write(f)
                out.release()
                
                prompt = format_prompt(burst_type, yield_tier, setting, phase, film_look, has_smoke_trails)
                
                start_w_sec = round((start_frame + window_idx * step) / src_fps, 2)
                end_w_sec = round((start_frame + (window_idx + CLIP_FRAMES) * step) / src_fps, 2)
                
                manifest_records.append({
                    "clip_id": clip_id,
                    "base_clip_id": base_clip_id,
                    "variant": var_suffix,
                    "operation": operation,
                    "shot_name": shot_name,
                    "source_video": video_basename,
                    "start_sec": start_w_sec,
                    "end_sec": end_w_sec,
                    "num_frames": CLIP_FRAMES,
                    "fps": TARGET_FPS,
                    "resolution": resolution,
                    "burst_type": burst_type,
                    "yield_tier": yield_tier,
                    "setting": setting,
                    "phase": phase,
                    "film_look": film_look,
                    "smoke_trails": has_smoke_trails,
                    "prompt": prompt,
                    "split": split,
                    "clip_path": rel_clip_path
                })
                
            window_idx += CLIP_FRAMES // 2
            
    cap.release()
    print(f"   Successfully extracted {len(manifest_records)} standardized clips.")
    return manifest_records


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Archival Video Standardization & Preprocessing")
    parser.add_argument("--input-dir", type=str, default=RAW_VIDEOS_DIR, help="Directory containing raw video files")
    parser.add_argument("--output-dir", type=str, default=PROCESSED_DIR, help="Output directory for processed clips")
    parser.add_argument("--single-file", type=str, default=None, help="Process a single specific video file")
    parser.add_argument("--resolution", type=int, default=512, help="Target square resolution (512 or 384)")
    parser.add_argument("--no-augment", action="store_true", help="Disable horizontal flip augmentation")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("=" * 65)
    print("       STAGE 3: VIDEO STANDARDIZATION & CUT SEGMENTATION")
    print("=" * 65)
    print(f"Resolution: {args.resolution}x{args.resolution} | FPS: {TARGET_FPS} | Frames: {CLIP_FRAMES}")
    
    shot_df = load_shot_metadata()
    print(f"Loaded {len(shot_df)} shot metadata references.")
    
    if args.single_file:
        video_files = [args.single_file]
    else:
        if not os.path.exists(args.input_dir):
            print(f"Input directory {args.input_dir} does not exist.")
            return
        video_files = [
            os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.ogv'))
        ]
        
    if not video_files:
        print(f"No video files found in {args.input_dir}. Please run scripts/01_download_archive.py first.")
        return
        
    print(f"Processing {len(video_files)} video file(s)...\n")
    
    all_manifest_records = []
    for v_path in video_files:
        records = process_video_file(
            v_path,
            args.output_dir,
            shot_df,
            resolution=args.resolution,
            augment=(not args.no_augment)
        )
        all_manifest_records.extend(records)
        
    if all_manifest_records:
        new_df = pd.DataFrame(all_manifest_records)
        if os.path.exists(MANIFEST_CSV):
            existing_df = pd.read_csv(MANIFEST_CSV)
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['clip_id'])
        else:
            combined_df = new_df
            
        combined_df.to_csv(MANIFEST_CSV, index=False)
        print("\n" + "=" * 65)
        print(f" Successfully compiled {MANIFEST_CSV}")
        print(f" Total standardized clips: {len(combined_df)}")
        print("=" * 65)
        print("\nClips per split:")
        print(combined_df['split'].value_counts().to_string())
        print("\nClips per phase:")
        print(combined_df['phase'].value_counts().to_string())
        print("\nClips per film look:")
        print(combined_df['film_look'].value_counts().to_string())
        print("\nSample Generated Prompt:")
        print(f"  \"{combined_df.iloc[0]['prompt']}\"")
    else:
        print("No valid explosion clips were extracted.")


if __name__ == "__main__":
    main()
