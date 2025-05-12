# agent_tools/color_presets.py
from typing import List, Dict

# --- Color Handling ---
# Approximate CIE XY coordinates for common color names.
COLOR_NAME_TO_XY: Dict[str, List[float]] = {
    # English
    "red": [0.701, 0.299], "strong red": [0.701, 0.299],
    "green": [0.172, 0.747], "lime": [0.215, 0.711],
    "blue": [0.136, 0.040], "strong blue": [0.136, 0.040],
    "yellow": [0.527, 0.413], "gold": [0.527, 0.413],
    "orange": [0.611, 0.375],
    "purple": [0.250, 0.100], "magenta": [0.409, 0.188], "pink": [0.415, 0.177],
    "cyan": [0.170, 0.361], "turquoise": [0.170, 0.361],
    "white": [0.313, 0.329], # Neutral white - Use color_temp for better control
    # Russian
    "красный": [0.701, 0.299], "алый": [0.701, 0.299],
    "зеленый": [0.172, 0.747], "салатовый": [0.215, 0.711],
    "синий": [0.136, 0.040], "голубой": [0.170, 0.361],
    "желтый": [0.527, 0.413], "золотой": [0.527, 0.413],
    "оранжевый": [0.611, 0.375], "рыжий": [0.611, 0.375],
    "фиолетовый": [0.250, 0.100], "пурпурный": [0.409, 0.188], "розовый": [0.415, 0.177],
    "бирюзовый": [0.170, 0.361],
    "белый": [0.313, 0.329],
}

# --- Color Temperature Preset Handling ---
# Mapping descriptive terms to Mired values (lower Mired = cooler Kelvin)
# Standard range often ~153 (6500K) to ~500 (2000K)
COLOR_TEMP_PRESETS: Dict[str, int] = {
    # English
    "coolest": 167,   # Approx 6500K
    "cool": 250,      # Approx 4000K
    "warm": 280,      # Approx 2500K
    "warmest": 333,   # Approx 2000K
    # Russian
    "самый холодный": 167,
    "холодный": 250,
    "белый дневной": 250, # Alias for cool
    "белый": 167,
    "теплый": 280,
    "самый теплый": 333,
}

def normalize_text_key(name: str) -> str:
    """Normalizes text key to lowercase and strips whitespace."""
    return name.lower().strip()