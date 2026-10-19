"""
scripts/02_parse_doe_metadata.py

Downloads (if necessary) and parses official atmospheric nuclear test records
from the U.S. Department of Energy Report DOE/NV-209 Rev 16:
'United States Nuclear Tests: July 1945 through September 1992'.

Extracts all atmospheric, underwater, and surface tests (1945-1963) and maps
them into the project's standardized conditioning schema:
- burst_type: underwater, surface-tower, air-burst
- yield_tier: sub-kiloton, tactical-yield, intermediate-yield, megaton-class
- setting: desert-day, desert-night, tropical-ocean, tropical-dawn

Outputs:
- data/doe_nv209_atmospheric_shots.csv
"""

import os
import re
import requests
import pandas as pd
from datetime import datetime
import pypdf

DOE_PDF_URL = "https://nnss.gov/wp-content/uploads/2023/08/DOE_NV-209_Rev16.pdf"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PDF_PATH = os.path.join(DATA_DIR, "DOE_NV-209_Rev16.pdf")
OUTPUT_CSV = os.path.join(DATA_DIR, "doe_nv209_atmospheric_shots.csv")


def ensure_pdf_downloaded():
    """Ensure the official DOE/NV-209 Rev 16 PDF is downloaded."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PDF_PATH):
        print(f"[1/4] Downloading DOE/NV-209 Rev 16 from {DOE_PDF_URL}...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(DOE_PDF_URL, headers=headers, timeout=60)
        resp.raise_for_status()
        with open(PDF_PATH, "wb") as f:
            f.write(resp.content)
        print(f"      Downloaded {len(resp.content):,} bytes to {PDF_PATH}")
    else:
        print(f"[1/4] Found existing PDF at {PDF_PATH} ({os.path.getsize(PDF_PATH):,} bytes)")


def parse_yield_kt(yield_str):
    """
    Parse official yield strings into numeric kilotons and categorical tier.
    Examples:
    - '21 kt' -> 21.0
    - '15 Mt' -> 15000.0
    - 'Less than 0.1 kt' -> 0.05
    - '1.2 kt' -> 1.2
    - 'Submegaton' -> 500.0
    - 'Low' -> 10.0
    - 'Zero' -> 0.0
    - '1 to 20 kt' -> 10.0 (midpoint)
    """
    if not yield_str or pd.isna(yield_str):
        return None, "tactical-yield"
    
    s = str(yield_str).strip()
    s_lower = s.lower()
    val_kt = None
    
    # Check for Megatons first
    m_mt = re.search(r'([\d\.]+)\s*(?:to\s*([\d\.]+))?\s*mt', s, re.IGNORECASE)
    if m_mt:
        if m_mt.group(2):
            val_kt = ((float(m_mt.group(1)) + float(m_mt.group(2))) / 2.0) * 1000.0
        else:
            val_kt = float(m_mt.group(1)) * 1000.0
    elif 'submegaton' in s_lower:
        val_kt = 500.0
    elif 'megaton' in s_lower:
        val_kt = 1000.0
    elif 'less than' in s_lower:
        m = re.search(r'([\d\.]+)', s)
        val_kt = float(m.group(1)) * 0.5 if m else 0.5
    elif 'zero' in s_lower:
        val_kt = 0.0
    else:
        # Check for kt
        m_range = re.search(r'([\d\.]+)\s*to\s*([\d\.]+)\s*kt', s, re.IGNORECASE)
        if m_range:
            val_kt = (float(m_range.group(1)) + float(m_range.group(2))) / 2.0
        else:
            m_kt = re.search(r'([\d\.]+)\s*kt', s, re.IGNORECASE)
            if m_kt:
                val_kt = float(m_kt.group(1))
            else:
                m_num = re.search(r'([\d\.]+)', s)
                if m_num:
                    val_kt = float(m_num.group(1))
                elif 'low' in s_lower:
                    val_kt = 10.0
                elif 'medium' in s_lower:
                    val_kt = 50.0

    # Categorize into 4 locked yield tiers
    if val_kt is None:
        tier = "tactical-yield"
    elif val_kt < 1.0:
        tier = "sub-kiloton"
    elif val_kt <= 20.0:
        tier = "tactical-yield"
    elif val_kt <= 1000.0:
        tier = "intermediate-yield"
    else:
        tier = "megaton-class"
        
    return val_kt, tier


def determine_setting(location_str, time_gmt_str):
    """
    Determine geographic and lighting setting:
    - tropical-ocean (Pacific: Bikini, Enewetak, Johnston Island, Christmas Island)
    - tropical-dawn (Pacific morning twilight detonations)
    - desert-day (Nevada / Alamogordo daytime detonations)
    - desert-night (Nevada pre-dawn / night detonations)
    """
    loc_lower = str(location_str).lower()
    is_pacific = any(k in loc_lower for k in [
        'bikini', 'enewetak', 'johnston', 'christmas', 'pacific', 'offshore'
    ])
    
    local_hour = 12
    try:
        parts = time_gmt_str.split(':')
        gmt_hour = int(parts[0])
        if is_pacific:
            # Marshall Islands local time = UTC + 12
            local_hour = (gmt_hour + 12) % 24
        else:
            # Nevada is Pacific Time (UTC - 8)
            local_hour = (gmt_hour - 8) % 24
    except Exception:
        pass
        
    if is_pacific:
        if 5 <= local_hour < 7:
            return "tropical-dawn"
        else:
            return "tropical-ocean"
    else:
        # Desert tests: Night is 20:00 to 06:00
        if local_hour < 6 or local_hour >= 20:
            return "desert-night"
        else:
            return "desert-day"


def map_burst_type(raw_type_str):
    """
    Maps DOE raw type to project burst type:
    - underwater: Underwater
    - surface-tower: Tower, Surface, Barge, Crater
    - air-burst: Airdrop, Airburst, Balloon, Rocket
    - underground: Shaft, Tunnel (filtered out)
    """
    t = str(raw_type_str).strip().lower()
    if 'underwater' in t:
        return 'underwater'
    elif any(k in t for k in ['tower', 'surface', 'barge', 'crater']):
        return 'surface-tower'
    elif any(k in t for k in ['airdrop', 'airburst', 'balloon', 'rocket']):
        return 'air-burst'
    elif any(k in t for k in ['shaft', 'tunnel', 'underground']):
        return 'underground'
    return 'surface-tower'


def parse_pdf():
    """Parse pages 28-44 of DOE/NV-209 Rev 16."""
    ensure_pdf_downloaded()
    print("[2/4] Parsing atmospheric and underwater tests from PDF...")
    
    reader = pypdf.PdfReader(PDF_PATH)
    
    # 1. Parse right column entries (Time, Lat, Lon, Elev, HOB, Purpose, Yield, Test#)
    right_re = re.compile(
        r'^(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+'      # Time GMT
        r'([-\d\.]+)\s+'                           # Lat
        r'([-\d\.]+)\s+'                           # Lon
        r'([-\d,]+)\s+'                            # Elevation ft
        r'([-\d,~A-Za-z\s]+?)\s+'                  # Height of burst / depth
        r'(Weapons Related|Weapons Effects|Safety Experiment|Vela Uniform|Plowshare|Joint US-UK|DoD Sponsored)\s+' # Purpose
        r'([0-9\.<>~A-Za-z\s\-]+?)\s+'             # Yield
        r'(\d+)$'                                  # Test Number
    )
    
    right_dict = {}
    for page_idx in range(28, 44):
        txt = reader.pages[page_idx].extract_text() or ''
        for line in txt.splitlines():
            s = line.strip()
            m = right_re.match(s)
            if m:
                test_num = int(m.group(8))
                lat_str = m.group(2).replace('--', '-')
                lon_str = m.group(3).replace('--', '-')
                right_dict[test_num] = {
                    'time_gmt': m.group(1),
                    'latitude': float(lat_str),
                    'longitude': float(lon_str),
                    'elevation_ft': m.group(4).replace(',', ''),
                    'burst_height_ft': m.group(5).strip(),
                    'purpose': m.group(6),
                    'yield_raw': m.group(7).strip(),
                }
    
    print(f"      Extracted {len(right_dict)} right-column test records.")
    
    # 2. Parse left column entries (Test#, Shot Name, Date, Sponsor, Location, Type)
    left_dict = {}
    current_op = "Trinity"
    
    for page_idx in range(28, 44):
        txt = reader.pages[page_idx].extract_text() or ''
        lines = txt.splitlines()
        
        # Determine the line index of the last timestamp on the page
        time_indices = [i for i, l in enumerate(lines) if re.match(r'^\d{2}:\d{2}:\d{2}', l.strip())]
        if not time_indices:
            continue
        last_time_idx = max(time_indices)
        left_lines = lines[last_time_idx + 1:]
        
        buffer = []
        for line in left_lines:
            s = line.strip()
            if not s:
                continue
                
            # Handle operation headers
            if 'Operation ' in s:
                m_date_test = re.search(r'(\d{2}/\d{2}/\d{4})', s)
                if not m_date_test:
                    op_clean = s[s.find('Operation '):].replace('– Continued', '').replace('- Continued', '').strip()
                    current_op = op_clean.replace('Operation ', '').strip()
                    continue
                else:
                    m_op = re.search(r'Operation\s+([A-Za-z0-9\-]+)', s)
                    if m_op:
                        current_op = m_op.group(1)
                    s = re.sub(r'Operation\s+[A-Za-z0-9\-]+', ' ', s).strip()
                    
            # Check for date line (mm/dd/yyyy)
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', s)
            if date_match:
                date_str = date_match.group(1)
                before_date = s[:date_match.start()].strip()
                after_date = s[date_match.end():].strip()
                
                header_tokens = buffer + ([before_date] if before_date else [])
                buffer = []
                full_header = " ".join(header_tokens).strip()
                
                # Match test number from full_header
                m_num = re.search(r'^\s*(\d+)\b', full_header)
                if not m_num:
                    m_num = re.search(r'\b(\d+)\b', full_header)
                
                if m_num:
                    t_num = int(m_num.group(1))
                    raw_name = full_header[m_num.end():].strip()
                    
                    # Clean shot name
                    shot_name = raw_name.split()[0] if raw_name else f"Shot_{t_num}"
                    shot_name = re.sub(r'[^A-Za-z0-9\-_]', '', shot_name)
                    if not shot_name:
                        shot_name = f"Shot_{t_num}"
                        
                    tokens = after_date.split()
                    raw_type = tokens[-1] if tokens else "Unknown"
                    sponsor = tokens[0] if tokens else "Unknown"
                    location = " ".join(tokens[1:-1]) if len(tokens) > 2 else "Unknown"
                    location = location.replace('- - - -', '').strip()
                    if not location:
                        location = "Pacific Proving Grounds" if any(k in after_date for k in ["Atoll", "Island"]) else "Nevada Test Site"
                        
                    left_dict[t_num] = {
                        'operation': current_op,
                        'shot_name': shot_name,
                        'date': date_str,
                        'sponsor': sponsor,
                        'location': location,
                        'raw_type': raw_type
                    }
            else:
                if not any(s.startswith(k) for k in ['Test', 'Date', '(mm/dd', '(GMT)', 'Sponsor', 'Location', 'Hole', 'Type']):
                    buffer.append(s)
                    
    print(f"      Extracted {len(left_dict)} left-column test records.")
    
    # 3. Combine matching tests
    all_test_nums = sorted(set(left_dict.keys()) & set(right_dict.keys()))
    print(f"[3/4] Successfully matched {len(all_test_nums)} tests between both columns.")
    
    records = []
    for t_num in all_test_nums:
        l = left_dict[t_num]
        r = right_dict[t_num]
        
        raw_type = l['raw_type']
        burst_type = map_burst_type(raw_type)
        
        # Filter out underground tests (Shaft, Tunnel)
        if burst_type == 'underground':
            continue
            
        yield_kt, yield_tier = parse_yield_kt(r['yield_raw'])
        setting = determine_setting(l['location'], r['time_gmt'])
        
        try:
            dt = datetime.strptime(l['date'], "%m/%d/%Y")
            iso_date = dt.strftime("%Y-%m-%d")
        except Exception:
            iso_date = l['date']
            
        records.append({
            'test_number': t_num,
            'operation': l['operation'],
            'shot_name': l['shot_name'],
            'date': iso_date,
            'time_gmt': r['time_gmt'],
            'sponsor': l['sponsor'],
            'location': l['location'],
            'raw_type': raw_type,
            'burst_type': burst_type,
            'yield_raw': r['yield_raw'],
            'yield_kt': yield_kt,
            'yield_tier': yield_tier,
            'setting': setting,
            'latitude': r['latitude'],
            'longitude': r['longitude'],
            'elevation_ft': r['elevation_ft'],
            'burst_height_ft': r['burst_height_ft'],
            'purpose': r['purpose']
        })
        
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[4/4] Saved {len(df)} atmospheric/surface/underwater shots to {OUTPUT_CSV}")
    
    # Print summary statistics
    print("\n=======================================================")
    print("       DOE/NV-209 ATMOSPHERIC SHOTS SUMMARY")
    print("=======================================================")
    print(f"Total atmospheric & underwater test shots: {len(df)}")
    print("\n[Burst Type Distribution]")
    print(df['burst_type'].value_counts().to_string())
    print("\n[Yield Tier Distribution]")
    print(df['yield_tier'].value_counts().to_string())
    print("\n[Setting Distribution]")
    print(df['setting'].value_counts().to_string())
    print("\n[Underwater Shots (Rare Class Audit)]")
    underwater_df = df[df['burst_type'] == 'underwater']
    for _, row in underwater_df.iterrows():
        print(f"  Test #{row['test_number']:3d}: {row['operation']:<16} {row['shot_name']:<12} ({row['date']}) - {row['yield_raw']} @ {row['location']}")


if __name__ == "__main__":
    parse_pdf()
