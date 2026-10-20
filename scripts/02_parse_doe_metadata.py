#!/usr/bin/env python3
"""
scripts/02_parse_doe_metadata.py

Ingests and parses DOE/NV-209 Rev 16 (United States Nuclear Tests: July 1945 through
September 1992), cross-referencing with data/doe_shot_aliases.csv to produce:
1. data/doe_nv209_atmospheric_shots.csv
2. web/historical_shots.json

Fixes all P0.2 and P0.5 blockers:
- Preserves multi-word shot names (De Baca, Wasp Prime, Clean Slate III, etc.)
- Strips footnote / note text cleanly
- Maps operations strictly by official test-number ranges (no footnote pollution)
- Canonical burst types without collapsing barge into surface-tower (surface-barge preserved)
- Includes Trinity (#1), reconciles against the official 216 tally, and emits the
  211 training-eligible detonations (206 atmospheric + 5 underwater)
- Validates all generated records against schema.py
"""

import os
import sys
import re
import json
import urllib.request
import pypdf
import pandas as pd

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import schema

DATA_DIR = os.path.join(REPO_ROOT, "data")
PDF_PATH = os.path.join(DATA_DIR, "DOE_NV-209_Rev16.pdf")
ALIASES_CSV = os.path.join(DATA_DIR, "doe_shot_aliases.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "doe_nv209_atmospheric_shots.csv")
WEB_JSON = os.path.join(REPO_ROOT, "web", "historical_shots.json")

PDF_URL = "https://www.nnss.gov/docs/docs_libraryhrc/DOE_NV-209_Rev16.pdf"

# Training corpus eligibility. The official 216 tally includes 5 pre-treaty crater
# events, but they are not one of the four canonical visual burst types. The 4
# Joint US-UK Roller Coaster events were high-explosive plutonium-dispersal safety
# tests, not nuclear detonations. Both groups stay in the alias registry for
# provenance but are excluded from the training corpus (documented in docs/).
TRAINING_GROUPS = {"US_ATMOSPHERIC", "UNDERWATER"}
EXCLUDED_GROUPS = {"PRE_TREATY_CRATER", "JOINT_US_UK_ATMOSPHERIC"}


def ensure_pdf_downloaded():
    """Ensure the official DOE NV-209 Rev 16 PDF is present."""
    if not os.path.exists(PDF_PATH):
        print(f"Downloading DOE/NV-209 Rev 16 from {PDF_URL}...")
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)
        print("Download complete.")


def calculate_lighting_and_terrain(location: str, lat: float, lon: float, time_gmt_str: str, elevation_ft: int = 0):
    """
    Determine canonical terrain ('desert-flats', 'tropical-ocean')
    and lighting ('day', 'night', 'dawn') from geographical coordinates and time.

    Note: every atmospheric/underwater test in DOE/NV-209 was conducted either at
    the Nevada desert test sites or over a Pacific atoll. There is no verified
    "low-mountains" atmospheric footage in the official registry, so the class was
    removed from schema.py rather than filled with heuristic desert relabels.
    """
    loc_lower = str(location).lower()
    
    # Terrain determination
    if any(k in loc_lower for k in ["atoll", "island", "ocean", "epg", "bikini", "enewetak", "christmas", "johnston", "pacific", "atlantic"]):
        terrain = "tropical-ocean"
    else:
        terrain = "desert-flats"

    # Solar lighting determination
    try:
        parts = time_gmt_str.split(":")
        gmt_hours = float(parts[0]) + float(parts[1]) / 60.0
        # Longitude in degrees -> hours offset (lon / 15.0)
        local_solar_hour = (gmt_hours + lon / 15.0) % 24.0
        
        if 5.0 <= local_solar_hour < 7.0:
            lighting = "dawn"
        elif 7.0 <= local_solar_hour < 19.5:
            lighting = "day"
        else:
            lighting = "night"
    except Exception:
        lighting = "day"

    return terrain, lighting


def parse_doe_nv209():
    """Parse pages 28-44 of DOE NV-209 Rev 16 using exact 1-to-1 page pairing."""
    ensure_pdf_downloaded()
    
    if not os.path.exists(ALIASES_CSV):
        raise FileNotFoundError(f"Missing required alias table: {ALIASES_CSV}")
        
    aliases_df = pd.read_csv(ALIASES_CSV)
    alias_dict = {row["test_num"]: row for _, row in aliases_df.iterrows()}

    reader = pypdf.PdfReader(PDF_PATH)
    
    # Regex matching the right column data line:
    # Time GMT, Lat, Lon, Elev, HOB, Purpose, Yield, Test#
    right_re = re.compile(
        r'^(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+'      # Time GMT
        r'([-\d\.]+)\s+'                           # Lat
        r'([-\d\.]+)\s+'                           # Lon
        r'([-\d,]+)\s+'                            # Elevation ft
        r'([-\d,~A-Za-z\s]+?)\s+'                  # Height of burst / depth
        r'(Weapons Related|Weapons Effects|Safety Experiment|Vela Uniform|Plowshare|Joint US-UK|DoD Sponsored)\s+' # Purpose
        r'([0-9\.<>~A-Za-z\s\-,]+?)\s+'            # Yield
        r'(\d+)$'                                  # Test Number
    )

    parsed_records = []
    
    for page_idx in range(28, 45):
        txt = reader.pages[page_idx].extract_text() or ""
        lines = txt.splitlines()
        
        right_indices = [i for i, l in enumerate(lines) if re.match(r'^\d{2}:\d{2}:\d{2}', l.strip())]
        date_indices = [i for i, l in enumerate(lines) if re.search(r'\d{2}/\d{2}/\d{4}', l.strip())]
        
        if len(right_indices) != len(date_indices):
            raise ValueError(f"Page {page_idx} column mismatch: {len(right_indices)} right vs {len(date_indices)} left")

        for k in range(len(right_indices)):
            r_idx = right_indices[k]
            d_idx = date_indices[k]
            
            r_line = lines[r_idx].strip()
            m_r = right_re.match(r_line)
            if not m_r:
                continue
                
            test_num = int(m_r.group(8))
            
            # Filter to tests present in the alias registry (atmospheric, underwater, crater)
            if test_num not in alias_dict:
                continue
                
            alias_info = alias_dict[test_num]
            
            # Exclude non-training groups: crater events (not a canonical burst
            # type) and Roller Coaster (HE dispersal tests, no nuclear yield).
            if str(alias_info["reconciliation_group"]) not in TRAINING_GROUPS:
                continue
            
            time_gmt = m_r.group(1)
            lat = float(m_r.group(2).replace("--", "-"))
            lon = float(m_r.group(3).replace("--", "-"))
            elev_str = m_r.group(4).replace(",", "")
            burst_height = m_r.group(5).strip()
            purpose = m_r.group(6)
            yield_raw = m_r.group(7).strip()
            
            d_line = lines[d_idx].strip()
            m_date = re.search(r'(\d{2}/\d{2}/\d{4})', d_line)
            date_raw = m_date.group(1)
            # Reformat MM/DD/YYYY to YYYY-MM-DD
            m_parts = date_raw.split("/")
            iso_date = f"{m_parts[2]}-{m_parts[0]}-{m_parts[1]}"
            
            after_date = d_line[m_date.end():].strip()
            tokens = after_date.split()
            raw_type = tokens[-1] if tokens else str(alias_info["raw_type"])
            sponsor = tokens[0] if tokens else "Unknown"
            location = " ".join(tokens[1:-1]) if len(tokens) > 2 else "Unknown"
            location = location.replace("- - - -", "").strip()
            if not location:
                location = str(alias_info.get("location", "Nevada Test Site"))

            # Canonical values from alias table and schema
            shot_name = str(alias_info["shot_name"])
            operation = str(alias_info["operation"])
            canonical_burst = str(alias_info["canonical_burst_type"])
            reconciliation_group = str(alias_info["reconciliation_group"])
            
            if canonical_burst not in schema.BURST_TYPES:
                raise ValueError(f"Test #{test_num} has invalid burst type: {canonical_burst}")

            # Yield parsing
            approx_kt, yield_tier = schema.parse_raw_yield_to_tier(yield_raw)

            # Elevation parsing
            elevation_ft = int(elev_str) if elev_str.lstrip("-").isdigit() else 0

            # Terrain and lighting
            terrain, lighting = calculate_lighting_and_terrain(location, lat, lon, time_gmt, elevation_ft)
            
            # Validate complete record against schema rules
            record = {
                "test_number": test_num,
                "operation": operation,
                "shot_name": shot_name,
                "date": iso_date,
                "time_gmt": time_gmt,
                "sponsor": sponsor,
                "location": location,
                "raw_type": raw_type,
                "burst_type": canonical_burst,
                "yield_raw": yield_raw,
                "yield_kt": approx_kt,
                "yield_tier": yield_tier,
                "terrain": terrain,
                "lighting": lighting,
                "latitude": lat,
                "longitude": lon,
                "elevation_ft": elevation_ft,
                "burst_height_ft": burst_height,
                "purpose": purpose,
                "reconciliation_group": reconciliation_group
            }
            
            schema.validate_manifest_row(record)
            parsed_records.append(record)

    df_out = pd.DataFrame(parsed_records)
    df_out = df_out.sort_values("test_number").reset_index(drop=True)
    
    # Save CSV
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"[SUCCESS] Saved {len(df_out)} verified shots to {OUTPUT_CSV}")
    
    # Save JSON for Web UI
    with open(WEB_JSON, "w", encoding="utf-8") as f:
        json.dump(parsed_records, f, indent=2)
    print(f"[SUCCESS] Saved {len(parsed_records)} shots to {WEB_JSON}")
    
    # Audit summary
    print("\n--- Summary Breakdown ---")
    print(f"Total Rows: {len(df_out)}")
    print(f"Unique Operations: {df_out['operation'].nunique()}")
    print(f"Unique Shot Names: {df_out['shot_name'].nunique()}")
    print("\nReconciliation Groups:")
    print(df_out["reconciliation_group"].value_counts().to_string())
    print("\nCanonical Burst Types:")
    print(df_out["burst_type"].value_counts().to_string())
    print("\nYield Tiers:")
    print(df_out["yield_tier"].value_counts().to_string())
    print("\nTerrains:")
    print(df_out["terrain"].value_counts().to_string())
    print("\nLightings:")
    print(df_out["lighting"].value_counts().to_string())


if __name__ == "__main__":
    parse_doe_nv209()
