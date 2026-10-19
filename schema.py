"""
schema.py - Canonical parameter schema, token validation, and prompt formatting
for the Parameter-Controlled Archival Nuclear Test Video Generator.

Non-negotiable rule (AGENTS.md §1 & §2 P0.5):
One schema only. Every prompt token produced by Stage 3 must be a valid
inference token and classifier class. Validate or fail.
"""

from typing import Dict, Any, Tuple, Optional
import re

# ==============================================================================
# Canonical Parameter Vocabularies
# ==============================================================================

BURST_TYPES = [
    "air-burst",
    "surface-tower",
    "surface-barge",
    "underwater"
]

YIELD_TIERS = [
    "sub-kiloton",       # < 1 kt
    "tactical-yield",    # 1 - 20 kt
    "intermediate-yield",# 20 - 1,000 kt
    "megaton-class"      # >= 1,000 kt (1 Mt+)
]

TERRAINS = [
    "desert-flats",
    "tropical-ocean",
    "low-mountains"
]

LIGHTINGS = [
    "day",
    "night",
    "dawn"
]

PHASES = [
    "early-fireball",
    "expansion",
    "stabilized-cloud"
]

FILM_LOOKS = [
    "black-and-white",
    "faded-technicolor",
    "colour-restored"
]

SMOKE_TRAILS_TOKENS = {
    True: "with smoke-trails",
    False: "without smoke-trails"
}

# ==============================================================================
# Yield Mapping Helper
# ==============================================================================

def yield_kt_to_tier(yield_kt: float) -> str:
    """Map numeric yield in kilotons to canonical yield tier."""
    if yield_kt < 1.0:
        return "sub-kiloton"
    elif yield_kt <= 20.0:
        return "tactical-yield"
    elif yield_kt < 1000.0:
        return "intermediate-yield"
    else:
        return "megaton-class"

def parse_raw_yield_to_tier(raw_yield: str) -> Tuple[Optional[float], str]:
    """
    Parse raw DOE yield strings (e.g. '21 kt', '< 0.1 kt', '15 Mt', '200 to 1,000 kt', 'Zero')
    Returns (approx_kt, yield_tier).
    """
    s = str(raw_yield).strip().lower().replace(',', '')
    if not s or s in ['zero', 'no yield', 'slight']:
        return 0.0, "sub-kiloton"
    if 'subkiloton' in s or 'sub-kiloton' in s or '< 1 kt' in s or '< 0.1 kt' in s:
        return 0.1, "sub-kiloton"
    if 'low' in s:
        return 5.0, "tactical-yield"
    if 'intermediate' in s:
        return 50.0, "intermediate-yield"
    if 'megaton' in s:
        return 1000.0, "megaton-class"

    # Match numeric ranges like '200 to 1000 kt' or '1 to 2 mt'
    m_range = re.search(r'([\d\.]+)\s*(?:to|-)\s*([\d\.]+)\s*(kt|mt)?', s)
    if m_range:
        v1 = float(m_range.group(1))
        v2 = float(m_range.group(2))
        avg = (v1 + v2) / 2.0
        unit = m_range.group(3) or ('mt' if 'mt' in s else 'kt')
        kt = avg * 1000.0 if unit == 'mt' else avg
        return kt, yield_kt_to_tier(kt)

    # Match single numbers like '21 kt', '15 mt', '0.002 kt'
    m_val = re.search(r'([<~>]?\s*[\d\.]+)\s*(kt|mt)?', s)
    if m_val:
        num_str = re.sub(r'[<~>\s]', '', m_val.group(1))
        val = float(num_str)
        unit = m_val.group(2) or ('mt' if 'mt' in s else 'kt')
        kt = val * 1000.0 if unit == 'mt' else val
        return kt, yield_kt_to_tier(kt)

    raise ValueError(f"Unable to parse yield string: '{raw_yield}'")

# ==============================================================================
# Setting Normalizer (for backwards compatibility)
# ==============================================================================

def normalize_terrain_and_lighting(setting_or_terrain: str, lighting: Optional[str] = None) -> Tuple[str, str]:
    """Resolve terrain and lighting, splitting combined setting tokens if needed."""
    s = setting_or_terrain.strip().lower()
    if s == "desert-day":
        return "desert-flats", "day"
    elif s == "desert-night":
        return "desert-flats", "night"
    elif s == "desert-dawn":
        return "desert-flats", "dawn"
    elif s == "tropical-ocean":
        return "tropical-ocean", lighting if lighting in LIGHTINGS else "day"
    elif s == "tropical-dawn":
        return "tropical-ocean", "dawn"
    elif s in TERRAINS:
        t = s
        l = lighting if lighting in LIGHTINGS else "day"
        return t, l
    else:
        raise ValueError(f"Unknown terrain/setting: '{setting_or_terrain}'. Must be one of {TERRAINS}")

# ==============================================================================
# Prompt Formatting and Validation
# ==============================================================================

def format_prompt(
    burst_type: str,
    yield_tier: str,
    terrain: str,
    lighting: str,
    phase: str,
    film_look: str,
    smoke_trails: bool = False
) -> str:
    """Format standardized conditioning prompt string from validated components."""
    if burst_type not in BURST_TYPES:
        raise ValueError(f"Invalid burst_type: '{burst_type}'. Must be one of {BURST_TYPES}")
    if yield_tier not in YIELD_TIERS:
        raise ValueError(f"Invalid yield_tier: '{yield_tier}'. Must be one of {YIELD_TIERS}")
    if terrain not in TERRAINS:
        raise ValueError(f"Invalid terrain: '{terrain}'. Must be one of {TERRAINS}")
    if lighting not in LIGHTINGS:
        raise ValueError(f"Invalid lighting: '{lighting}'. Must be one of {LIGHTINGS}")
    if phase not in PHASES:
        raise ValueError(f"Invalid phase: '{phase}'. Must be one of {PHASES}")
    if film_look not in FILM_LOOKS:
        raise ValueError(f"Invalid film_look: '{film_look}'. Must be one of {FILM_LOOKS}")
    if not isinstance(smoke_trails, (bool, int)):
        raise ValueError(f"smoke_trails must be boolean, got {type(smoke_trails)}")

    smoke_str = SMOKE_TRAILS_TOKENS[bool(smoke_trails)]
    return (
        f"{burst_type} nuclear detonation, {yield_tier}, {terrain}, {lighting}, "
        f"{phase} phase, {film_look} archival footage, {smoke_str}"
    )

def validate_prompt(prompt: str) -> Dict[str, Any]:
    """
    Validate and extract tokens from a standardized prompt string.
    Raises ValueError if prompt does not conform to the canonical schema.
    """
    p = prompt.strip()
    # Match pattern:
    # {burst_type} nuclear detonation, {yield_tier}, {terrain}, {lighting}, {phase} phase, {film_look} archival footage, {smoke_str}
    pattern = re.compile(
        r'^(?P<burst_type>[\w\-]+)\s+nuclear detonation,\s+'
        r'(?P<yield_tier>[\w\-]+),\s+'
        r'(?P<terrain>[\w\-]+),\s+'
        r'(?P<lighting>[\w\-]+),\s+'
        r'(?P<phase>[\w\-]+)\s+phase,\s+'
        r'(?P<film_look>[\w\-]+)\s+archival footage,\s+'
        r'(?P<smoke>with smoke-trails|without smoke-trails)$'
    )
    m = pattern.match(p)
    if not m:
        # Check if legacy 6-token pattern (terrain without lighting):
        legacy_pattern = re.compile(
            r'^(?P<burst_type>[\w\-]+)\s+nuclear detonation,\s+'
            r'(?P<yield_tier>[\w\-]+),\s+'
            r'(?P<setting>[\w\-]+),\s+'
            r'(?P<phase>[\w\-]+)\s+phase,\s+'
            r'(?P<film_look>[\w\-]+)\s+archival footage,\s+'
            r'(?P<smoke>(?:with diagnostic smoke-trails|with smoke-trails|without smoke-trails))$'
        )
        lm = legacy_pattern.match(p)
        if not lm:
            raise ValueError(f"Prompt does not match canonical template: '{prompt}'")
        burst_type = lm.group("burst_type")
        yield_tier = lm.group("yield_tier")
        setting = lm.group("setting")
        phase = lm.group("phase")
        film_look = lm.group("film_look")
        smoke_token = lm.group("smoke")
        terrain, lighting = normalize_terrain_and_lighting(setting)
        smoke_trails = "without" not in smoke_token
    else:
        burst_type = m.group("burst_type")
        yield_tier = m.group("yield_tier")
        terrain = m.group("terrain")
        lighting = m.group("lighting")
        phase = m.group("phase")
        film_look = m.group("film_look")
        smoke_trails = m.group("smoke") == "with smoke-trails"

    # Validate against canonical sets
    if burst_type not in BURST_TYPES:
        raise ValueError(f"Invalid burst_type '{burst_type}' in prompt. Allowed: {BURST_TYPES}")
    if yield_tier not in YIELD_TIERS:
        raise ValueError(f"Invalid yield_tier '{yield_tier}' in prompt. Allowed: {YIELD_TIERS}")
    if terrain not in TERRAINS:
        raise ValueError(f"Invalid terrain '{terrain}' in prompt. Allowed: {TERRAINS}")
    if lighting not in LIGHTINGS:
        raise ValueError(f"Invalid lighting '{lighting}' in prompt. Allowed: {LIGHTINGS}")
    if phase not in PHASES:
        raise ValueError(f"Invalid phase '{phase}' in prompt. Allowed: {PHASES}")
    if film_look not in FILM_LOOKS:
        raise ValueError(f"Invalid film_look '{film_look}' in prompt. Allowed: {FILM_LOOKS}")

    return {
        "burst_type": burst_type,
        "yield_tier": yield_tier,
        "terrain": terrain,
        "lighting": lighting,
        "phase": phase,
        "film_look": film_look,
        "smoke_trails": smoke_trails
    }

def validate_manifest_row(row: Dict[str, Any]) -> None:
    """Validate that a manifest or shot record conforms strictly to the canonical schema."""
    burst_type = row.get("burst_type")
    yield_tier = row.get("yield_tier")
    terrain = row.get("terrain")
    lighting = row.get("lighting")
    phase = row.get("phase")
    film_look = row.get("film_look")

    if burst_type not in BURST_TYPES:
        raise ValueError(f"Invalid burst_type '{burst_type}'. Allowed: {BURST_TYPES}")
    if yield_tier not in YIELD_TIERS:
        raise ValueError(f"Invalid yield_tier '{yield_tier}'. Allowed: {YIELD_TIERS}")
    if terrain not in TERRAINS:
        raise ValueError(f"Invalid terrain '{terrain}'. Allowed: {TERRAINS}")
    if lighting not in LIGHTINGS:
        raise ValueError(f"Invalid lighting '{lighting}'. Allowed: {LIGHTINGS}")
    if phase is not None and phase not in PHASES:
        raise ValueError(f"Invalid phase '{phase}'. Allowed: {PHASES}")
    if film_look is not None and film_look not in FILM_LOOKS:
        raise ValueError(f"Invalid film_look '{film_look}'. Allowed: {FILM_LOOKS}")
