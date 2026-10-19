"""
scripts/01_download_archive.py

Stage 1 Data Ingestion Script.
Discovers, catalogues, and downloads public-domain archival nuclear test films
from the Internet Archive (AEC, DoD, and Department of Energy collections).

Generates and maintains:
- data/sources.csv: Comprehensive source manifest tracking archive IDs, titles,
  operations, licenses, file sizes, direct download URLs, and download status.
- raw_videos/: Directory where raw video files are downloaded (gitignored).

Usage:
  python scripts/01_download_archive.py --sync-metadata
  python scripts/01_download_archive.py --download-sample
  python scripts/01_download_archive.py --operation Crossroads
  python scripts/01_download_archive.py --download-all
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "raw_videos")
SOURCES_CSV = os.path.join(DATA_DIR, "sources.csv")

# Pre-curated core official DOE / DoD / DTRA archival test film collection
CORE_ARCHIVE_ITEMS = [
    # 1. Project Trinity
    {"id": "gov.doe.0800001", "operation": "Trinity", "title": "Nuclear Test Film - Trinity Shot"},
    
    # 2. Operation Crossroads (Includes BAKER underwater shot!)
    {"id": "gov.doe.0800002", "operation": "Crossroads", "title": "Nuclear Test Film - Project Crossroads"},
    {"id": "gov.dod.dimoc.23521", "operation": "Crossroads", "title": "Operation Crossroads 1946 (DoD DIMOC)"},
    
    # 3. Operation Sandstone
    {"id": "gov.doe.0800003", "operation": "Sandstone", "title": "Nuclear Test Film - Operation Sandstone"},
    {"id": "gov.doe.0800008", "operation": "Sandstone", "title": "Nuclear Test Film - Operation Sandstone - Navy"},
    
    # 4. Operation Greenhouse
    {"id": "gov.doe.0800009", "operation": "Greenhouse", "title": "Nuclear Test Film - Operation Greenhouse"},
    
    # 5. Operation Ranger & Buster-Jangle
    {"id": "gov.doe.0800010", "operation": "Ranger / Buster-Jangle", "title": "Nuclear Test Film - Operation Ranger, Operation Buster/Jangle"},
    
    # 6. Operation Tumbler-Snapper
    {"id": "gov.doe.0800011", "operation": "Tumbler-Snapper", "title": "Nuclear Test Film - Operation Tumbler-Snapper"},
    
    # 7. Operation Ivy (First megaton-class thermonuclear test: MIKE)
    {"id": "gov.doe.0800012", "operation": "Ivy", "title": "Nuclear Test Film - Operation Ivy"},
    
    # 8. Operation Upshot-Knothole
    {"id": "gov.doe.0800015", "operation": "Upshot-Knothole", "title": "Nuclear Test Film - Operation Upshot-Knothole"},
    
    # 9. Operation Castle (High-yield thermonuclear series: BRAVO, ROMEO, YANKEE)
    {"id": "gov.doe.0800013", "operation": "Castle", "title": "Nuclear Test Film - Operation Castle"},
    {"id": "MilitaryEffectsStudiesonOperationCastle1954", "operation": "Castle", "title": "Military Effects Studies on Operation CASTLE"},
    
    # 10. Operation Teapot
    {"id": "gov.doe.0800017", "operation": "Teapot", "title": "Nuclear Test Film - Operation Teapot"},
    
    # 11. Operation Wigwam (Underwater deep ocean detonation!)
    {"id": "gov.doe.0800018", "operation": "Wigwam", "title": "Nuclear Test Film - Operation Wigwam"},
    
    # 12. Operation Redwing
    {"id": "gov.doe.0800019", "operation": "Redwing", "title": "Nuclear Test Film - Operation Redwing"},
    
    # 13. Operation Plumbbob
    {"id": "28272CivilDefenseMissionFallout", "operation": "Plumbbob", "title": "Operation Plumbbob 1957 - Mission Fallout"},
    
    # 14. Operation Hardtack I (Underwater tests WAHOO and UMBRELLA!)
    {"id": "OperationHARDTACK_UnderwaterTests1958", "operation": "Hardtack I", "title": "Operation HARDTACK Military Effects Studies: Underwater Tests"},
    {"id": "OperationHARDTACK_BasicEffectsStructuresandMateriel1958", "operation": "Hardtack I", "title": "Operation HARDTACK Basic Effects Structures and Materiel"},
    {"id": "OperationHARDTACK_HighAltitudeTests1958", "operation": "Hardtack I", "title": "Operation HARDTACK High Altitude Tests"},
    
    # 15. Operation Dominic I (Pacific 1962 series, including ASROC Swordfish underwater test)
    {"id": "OperationDOMINICNuclearTests1962", "operation": "Dominic", "title": "Operation DOMINIC Nuclear Tests 1962"},
    {"id": "DominicSunset", "operation": "Dominic", "title": "Operation Dominic - Sunset Air Burst"},
]


def fetch_archive_metadata(identifier):
    """Fetch item metadata and file listings from Internet Archive API."""
    url = f"https://archive.org/metadata/{identifier}"
    headers = {"User-Agent": "NuclearArchivalVideoResearch/1.0 (Educational/Scientific AI)"}
    
    try:
        resp = requests.get(url, headers=headers, timeout=25)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            print(f"Warning: Item {identifier} not found (404).")
            return None
        else:
            print(f"Warning: Failed to fetch {identifier} (Status {resp.status_code}).")
            return None
    except Exception as e:
        print(f"Error fetching metadata for {identifier}: {e}")
        return None


def select_best_video_file(files):
    """
    Select the highest quality clean MP4 video file from an item's file list.
    Prioritizes .mp4 files (h.264 / 512Kb MPEG4) over .ogv or raw scans.
    """
    mp4_files = [
        f for f in files 
        if f.get("name", "").lower().endswith(".mp4") and not f.get("name", "").startswith(".")
    ]
    
    if not mp4_files:
        # Fallback to .ogv or other video files
        mp4_files = [
            f for f in files 
            if any(f.get("name", "").lower().endswith(ext) for ext in [".ogv", ".mkv", ".avi", ".mov"])
        ]
        
    if not mp4_files:
        return None
        
    # Prefer files that are not tiny thumbnails or audio-only
    # Sort by size descending, but prefer MP4
    mp4_files.sort(key=lambda x: int(x.get("size", 0)), reverse=True)
    return mp4_files[0]


def sync_metadata():
    """Discover, verify provenance, and compile data/sources.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
    
    print("=" * 65)
    print("       STAGE 1: COMPILING ARCHIVAL VIDEO SOURCES MANIFEST")
    print("=" * 65)
    print(f"Querying Internet Archive for {len(CORE_ARCHIVE_ITEMS)} core historical reels...\n")
    
    records = []
    
    for item in CORE_ARCHIVE_ITEMS:
        ident = item["id"]
        op = item["operation"]
        print(f" -> Inspecting [{ident}] ({op})...")
        
        meta = fetch_archive_metadata(ident)
        if not meta:
            continue
            
        m_info = meta.get("metadata", {})
        files = meta.get("files", [])
        
        title = m_info.get("title", item["title"])
        creator = m_info.get("creator", "U.S. Department of Energy / Department of Defense")
        year = m_info.get("year", m_info.get("publicdate", "")[:4])
        license_url = m_info.get("licenseurl", "U.S. Federal Government Public Domain (17 U.S.C. § 105)")
        
        best_video = select_best_video_file(files)
        if not best_video:
            print(f"    [!] No suitable video file found for {ident}.")
            continue
            
        video_name = best_video.get("name")
        video_size = int(best_video.get("size", 0))
        video_size_mb = round(video_size / (1024 * 1024), 2)
        video_fmt = best_video.get("format", "MPEG4")
        
        download_url = f"https://archive.org/download/{ident}/{video_name}"
        local_filename = f"{op.replace(' ', '_').replace('/', '_')}_{ident}_{video_name}"
        local_path = os.path.join("raw_videos", local_filename)
        full_local_path = os.path.join(PROJECT_ROOT, local_path)
        
        # Check if already downloaded and verified
        downloaded = os.path.exists(full_local_path) and os.path.getsize(full_local_path) > 0
        
        records.append({
            "archive_id": ident,
            "operation": op,
            "title": title,
            "year": year,
            "creator": creator,
            "license": license_url,
            "video_filename": video_name,
            "download_url": download_url,
            "size_mb": video_size_mb,
            "format": video_fmt,
            "local_path": local_path,
            "downloaded": downloaded,
            "access_date": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
        
        # Friendly rate limiting to archive.org API
        time.sleep(0.3)
        
    df = pd.DataFrame(records)
    df.to_csv(SOURCES_CSV, index=False)
    
    print("\n" + "=" * 65)
    print(f" Successfully compiled {len(df)} archival video sources to:")
    print(f"   {SOURCES_CSV}")
    print("=" * 65)
    print(f"Total video archive volume: {df['size_mb'].sum():,.1f} MB (~{df['size_mb'].sum()/1024:.2f} GB)")
    print("\nBreakdown by Operation:")
    print(df['operation'].value_counts().to_string())
    
    return df


def download_file(url, target_path, expected_size_bytes=None):
    """Download a file with streaming progress bar and resume support."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    temp_path = target_path + ".tmp"
    existing_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
    
    headers = {"User-Agent": "NuclearArchivalVideoResearch/1.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        
    response = requests.get(url, headers=headers, stream=True, timeout=30)
    
    if response.status_code == 416:
        # Range not satisfiable -> file already complete
        if os.path.exists(temp_path):
            os.replace(temp_path, target_path)
            return True
            
    if response.status_code not in [200, 206]:
        print(f"Download failed with HTTP status {response.status_code}: {url}")
        return False
        
    total_size = int(response.headers.get("content-length", 0)) + existing_bytes
    mode = "ab" if existing_bytes > 0 else "wb"
    
    basename = os.path.basename(target_path)
    with open(temp_path, mode) as f, tqdm(
        desc=basename[:30],
        total=total_size,
        initial=existing_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
                
    os.replace(temp_path, target_path)
    return True


def run_downloads(filter_op=None, filter_id=None, sample_only=False, max_items=None):
    """Execute video downloads according to filters."""
    if not os.path.exists(SOURCES_CSV):
        print("sources.csv not found. Running metadata sync first...")
        df = sync_metadata()
    else:
        df = pd.read_csv(SOURCES_CSV)
        
    targets = df.copy()
    
    if filter_op:
        targets = targets[targets["operation"].str.contains(filter_op, case=False, na=False)]
    if filter_id:
        targets = targets[targets["archive_id"] == filter_id]
    if sample_only:
        # Pick 2 representative high-priority reels: Wigwam (underwater) and Crossroads
        sample_ids = ["gov.doe.0800018", "gov.doe.0800002"]
        targets = targets[targets["archive_id"].isin(sample_ids)]
    if max_items:
        targets = targets.head(max_items)
        
    if len(targets) == 0:
        print("No matching video sources found for specified filters.")
        return
        
    print(f"\nStarting download for {len(targets)} video reel(s) ({targets['size_mb'].sum():.1f} MB total)...\n")
    
    success_count = 0
    for idx, row in targets.iterrows():
        ident = row["archive_id"]
        title = row["title"]
        url = row["download_url"]
        local_rel = row["local_path"]
        target_path = os.path.join(PROJECT_ROOT, local_rel)
        
        print(f"\n[{idx+1}/{len(targets)}] {title}")
        print(f"       Archive ID: {ident} | Size: {row['size_mb']} MB")
        print(f"       Target: {local_rel}")
        
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print("       -> Already downloaded. Skipping.")
            df.loc[df["archive_id"] == ident, "downloaded"] = True
            success_count += 1
            continue
            
        success = download_file(url, target_path)
        if success:
            df.loc[df["archive_id"] == ident, "downloaded"] = True
            success_count += 1
            print(f"       -> Completed: {os.path.getsize(target_path):,} bytes.")
        else:
            print("       -> Download failed.")
            
    df.to_csv(SOURCES_CSV, index=False)
    print(f"\nDownload run finished. Successfully verified {success_count}/{len(targets)} video reels.")


def main():
    parser = argparse.ArgumentParser(description="Internet Archive Nuclear Test Ingestion Engine")
    parser.add_argument("--sync-metadata", action="store_true", help="Compile and refresh data/sources.csv metadata")
    parser.add_argument("--download-all", action="store_true", help="Download all reels in sources.csv")
    parser.add_argument("--download-sample", action="store_true", help="Download sample test reels (Wigwam & Crossroads)")
    parser.add_argument("--operation", type=str, help="Download only reels for a specific operation (e.g. Wigwam, Crossroads)")
    parser.add_argument("--id", type=str, help="Download a specific archive ID")
    parser.add_argument("--limit", type=int, help="Limit number of reels to download")
    
    args = parser.parse_args()
    
    # If no flags passed, default to sync-metadata
    if not any([args.sync_metadata, args.download_all, args.download_sample, args.operation, args.id]):
        print("No execution flag provided. Defaulting to --sync-metadata.")
        sync_metadata()
    elif args.sync_metadata:
        sync_metadata()
    else:
        run_downloads(
            filter_op=args.operation,
            filter_id=args.id,
            sample_only=args.download_sample,
            max_items=args.limit
        )


if __name__ == "__main__":
    main()
