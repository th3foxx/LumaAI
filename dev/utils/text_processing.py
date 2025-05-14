import logging
import re
from typing import List, Optional, Dict, Any # Added Dict, Any for _NLU_ATTR_TO_STANDARD

from thefuzz import process as fuzz_process, fuzz
import pymorphy3

# Import from local tools and color_presets - these are still relevant for some utils
from tools.color_presets import COLOR_NAME_TO_XY, COLOR_TEMP_PRESETS, normalize_text_key

logger = logging.getLogger(__name__)

# --- Pymorphy3 Analyzer (Initialize once here) ---
try:
    morph = pymorphy3.MorphAnalyzer()
    logger.info("Pymorphy3 MorphAnalyzer initialized in utils.")
except Exception as e:
    logger.error(f"Failed to initialize Pymorphy3 MorphAnalyzer in utils: {e}. Word normalization will be basic.")
    morph = None

# --- Helper Functions (from original offline_controller.py) ---

def find_best_match(query: str, choices: List[str], scorer=fuzz.token_sort_ratio, score_cutoff=75) -> Optional[str]:
    """
    Finds the best match for a query string from a list of choices using fuzzy matching.
    """
    if not choices:
        return None
    if not query: # Handle empty query to avoid errors with fuzz_process
        return None
    match = fuzz_process.extractOne(query, choices, scorer=scorer, score_cutoff=score_cutoff)
    return match[0] if match else None

def _normalize_russian_text_with_pymorphy(text: str) -> str:
    """
    Normalizes Russian text using Pymorphy3 (normal form of each word, lowercased).
    This function is crucial and used by NLU/OfflineProcessor.
    """
    if not text:
        return ""
    
    text_lower = text.lower()

    if not morph:
        return text_lower 
    try:
        words = text_lower.split()
        normalized_words = [morph.parse(word)[0].normal_form for word in words if word]
        return " ".join(normalized_words)
    except Exception as e:
        logger.warning(f"Pymorphy3 normalization failed for '{text}': {e}. Returning lowercased original.")
        return text_lower

# --- City to Timezone Mapping (Still useful if NLU engine uses it directly) ---
# This could also be part of a more specific "location_utils.py" if it grows.
_PRE_NORMALIZED_CITY_TO_TIMEZONE: Dict[str, str] = {}

def initialize_city_timezones():
    """
    Initializes the mapping of normalized city names to TZ database names.
    Called once at startup.
    """
    global _PRE_NORMALIZED_CITY_TO_TIMEZONE
    if _PRE_NORMALIZED_CITY_TO_TIMEZONE: # Avoid re-initialization
        return

    city_map_raw = {
        "москва": "Europe/Moscow", "мск": "Europe/Moscow",
        "санкт-петербург": "Europe/Moscow", "спб": "Europe/Moscow", "питер": "Europe/Moscow",
        "лондон": "Europe/London",
        "париж": "Europe/Paris",
        "нью-йорк": "America/New_York", "нью йорк": "America/New_York", "nyc": "America/New_York",
        "токио": "Asia/Tokyo",
        "берлин": "Europe/Berlin",
        # ... (include all your cities from the original file) ...
        "kyiv": "Europe/Kyiv",
    }
    temp_map = {}
    for city, tz in city_map_raw.items():
        # Normalize keys using the same function used for input text
        # Ensure _normalize_russian_text_with_pymorphy is defined before this call
        temp_map[_normalize_russian_text_with_pymorphy(city)] = tz 
    _PRE_NORMALIZED_CITY_TO_TIMEZONE = temp_map
    logger.info(f"Utilities: Initialized city to timezone map with {len(_PRE_NORMALIZED_CITY_TO_TIMEZONE)} entries.")

# Call initialization here so it's ready when module is imported
initialize_city_timezones()


# --- Standard Attribute Mapping (Still useful for NLU engine) ---
# This helps standardize attribute names coming from Rasa or other NLU.
# It's used by RasaNLUEngine.
_NLU_ATTR_TO_STANDARD: Dict[str, str] = {
    normalize_text_key("яркость"): "brightness",
    normalize_text_key("brightness"): "brightness",
    normalize_text_key("уровень яркости"): "brightness",
    normalize_text_key("освещенность"): "brightness",
    normalize_text_key("цвет"): "color",
    normalize_text_key("color"): "color",
    normalize_text_key("температура"): "color_temp",
    normalize_text_key("цветовая температура"): "color_temp",
    normalize_text_key("оттенок белого"): "color_temp",
    normalize_text_key("теплота света"): "color_temp",
    normalize_text_key("temperature"): "color_temp",
    normalize_text_key("состояние"): "state",
    normalize_text_key("state"): "state",
    normalize_text_key("режим"): "mode",
    normalize_text_key("mode"): "mode",
}

# --- Other constants from offline_controller that might be useful for NLU/OfflineProcessor ---
# These were global in offline_controller.py.
# RASA_NLU_URL, RASA_NLU_TIMEOUT, RASA_INTENT_CONFIDENCE_THRESHOLD
# are now part of settings.rasa_nlu and used by RasaNLUEngine directly from its config.
# So, no need to keep them here as global constants.

# The _last_mentioned_device global state is now managed by ConnectionManager.

# The functions parse_offline_command and execute_offline_command are now
# fully replaced by RasaNLUEngine and DefaultOfflineCommandProcessor.

# You might have other small utility functions in offline_controller.py.
# Review it carefully and move any reusable, stateless helper functions here.
# For example, if find_capability was a general utility, it could come here,
# but it seems specific to device_control tool.
