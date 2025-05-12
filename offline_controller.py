import logging
import json
import re # Added for regex matching of city names
from typing import Dict, Optional, List, Any

from thefuzz import process as fuzz_process, fuzz
import httpx
import pymorphy3

# Import from local tools and color_presets
from tools.device_control import find_capability # Retained
from tools.color_presets import COLOR_NAME_TO_XY, COLOR_TEMP_PRESETS, normalize_text_key
from mqtt_client import mqtt_client
from settings import settings
from tools.device_control import set_device_attribute
from tools.time import get_current_time # Import the time tool

logger = logging.getLogger(__name__)

# --- Pymorphy3 Analyzer ---
try:
    morph = pymorphy3.MorphAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize Pymorphy3 MorphAnalyzer: {e}. Word normalization will be basic.")
    morph = None

# --- State ---
_last_mentioned_device: Optional[str] = None

# --- Rasa NLU API Configuration ---
RASA_NLU_URL = settings.rasa_nlu.url
RASA_NLU_TIMEOUT = settings.rasa_nlu.timeout
RASA_INTENT_CONFIDENCE_THRESHOLD = settings.rasa_nlu.intent_confidence_threshold

# --- Standard Attribute Mapping (for devices) ---
_NLU_ATTR_TO_STANDARD = {
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

# --- Helper Functions ---

def find_best_match(query: str, choices: List[str], scorer=fuzz.token_sort_ratio, score_cutoff=75) -> Optional[str]:
    if not choices:
        return None
    match = fuzz_process.extractOne(query, choices, scorer=scorer, score_cutoff=score_cutoff)
    return match[0] if match else None

def _normalize_russian_text_with_pymorphy(text: str) -> str:
    """Normalizes Russian text using Pymorphy3 (normal form of each word, lowercased)."""
    if not text: # handle empty string input explicitly
        return ""
    
    text_lower = text.lower() # Always lowercase

    if not morph:
        return text_lower # Fallback if Pymorphy3 not initialized
    try:
        words = text_lower.split()
        # Filter out empty strings that might result from multiple spaces
        normalized_words = [morph.parse(word)[0].normal_form for word in words if word]
        return " ".join(normalized_words)
    except Exception as e:
        logger.warning(f"Pymorphy3 normalization failed for '{text}': {e}. Returning lowercased original.")
        return text_lower

# --- City to Timezone Mapping ---
_PRE_NORMALIZED_CITY_TO_TIMEZONE = {}

def initialize_city_timezones():
    global _PRE_NORMALIZED_CITY_TO_TIMEZONE
    # Raw city names (mostly lowercase, some common abbreviations) and their TZ Database names
    city_map_raw = {
        "москва": "Europe/Moscow", "мск": "Europe/Moscow",
        "санкт-петербург": "Europe/Moscow", "спб": "Europe/Moscow", "питер": "Europe/Moscow",
        "лондон": "Europe/London",
        "париж": "Europe/Paris",
        "нью-йорк": "America/New_York", "нью йорк": "America/New_York", "nyc": "America/New_York",
        "токио": "Asia/Tokyo",
        "берлин": "Europe/Berlin",
        "рим": "Europe/Rome",
        "сидней": "Australia/Sydney",
        "дубай": "Asia/Dubai",
        "пекин": "Asia/Shanghai", # Beijing is in China Standard Time (Asia/Shanghai)
        "киев": "Europe/Kyiv",
        "минск": "Europe/Minsk",
        "астана": "Asia/Almaty", # Kazakhstan unified to UTC+5 (Almaty time) in 2024
        "алматы": "Asia/Almaty",
        "екатеринбург": "Asia/Yekaterinburg", "екб": "Asia/Yekaterinburg",
        "новосибирск": "Asia/Novosibirsk", "нск": "Asia/Novosibirsk",
        "владивосток": "Asia/Vladivostok",
        "сан-франциско": "America/Los_Angeles", "сан франциско": "America/Los_Angeles",
        "чикаго": "America/Chicago",
        "калининград": "Europe/Kaliningrad",
        "самара": "Europe/Samara",
        "омск": "Asia/Omsk",
        "красноярск": "Asia/Krasnoyarsk",
        "иркутск": "Asia/Irkutsk",
        "якутск": "Asia/Yakutsk",
        "магадан": "Asia/Magadan",
        "камчатка": "Asia/Kamchatka", "петропавловск-камчатский": "Asia/Kamchatka",
        "стамбул": "Europe/Istanbul",
        "варшава": "Europe/Warsaw",
        "прага": "Europe/Prague",
        # Add common English names if users might say them
        "moscow": "Europe/Moscow",
        "london": "Europe/London",
        "paris": "Europe/Paris",
        "new york": "America/New_York",
        "tokyo": "Asia/Tokyo",
        "berlin": "Europe/Berlin",
        "kyiv": "Europe/Kyiv", # Modern spelling
    }
    temp_map = {}
    for city, tz in city_map_raw.items():
        # Normalize keys using the same function used for input text
        temp_map[_normalize_russian_text_with_pymorphy(city)] = tz
    _PRE_NORMALIZED_CITY_TO_TIMEZONE = temp_map
    logger.info(f"Initialized city to timezone map with {len(_PRE_NORMALIZED_CITY_TO_TIMEZONE)} entries.")

# Initialize the city timezone map after morph is potentially initialized
initialize_city_timezones()


async def parse_offline_command(text: str) -> Optional[Dict[str, Any]]:
    global _last_mentioned_device
    payload = {"text": text}
    # Initialize with None or default values
    parsed_command_data: Dict[str, Any] = {
        "intent": None,
        "device": None,
        "attribute": None,
        "raw_value": None,
        "timezone_str_for_tool": None, # For get_time intent
    }
    logger.debug(f"Attempting offline parse via Rasa NLU (URL: {RASA_NLU_URL}) for: '{text}'")

    if not RASA_NLU_URL:
        logger.error("Rasa NLU URL not configured. Cannot parse offline command.")
        return None

    try:
        async with httpx.AsyncClient(timeout=RASA_NLU_TIMEOUT) as client:
            response = await client.post(RASA_NLU_URL, json=payload)
            response.raise_for_status()
            result = response.json()
        logger.debug(f"Rasa NLU raw result: {json.dumps(result, indent=2, ensure_ascii=False)}")

        intent_data = result.get("intent", {})
        intent_name = intent_data.get("name")
        confidence = intent_data.get("confidence", 0.0)
        entities = result.get("entities", [])
        
        parsed_command_data["intent"] = intent_name

        if not intent_name or confidence < RASA_INTENT_CONFIDENCE_THRESHOLD:
            logger.warning(f"NLU confidence low ({confidence:.2f} < {RASA_INTENT_CONFIDENCE_THRESHOLD}) for intent '{intent_name}'. Text: '{text}'. Aborting.")
            return None

        logger.info(f"Rasa NLU classified intent as '{intent_name}' with confidence {confidence:.2f}")

        # --- Intent-Specific Parsing ---
        if intent_name == "get_time":
            timezone_entity = next((e for e in entities if e['entity'] == 'timezone_location'), None)
            resolved_timezone_arg = None

            if timezone_entity:
                user_specified_location = timezone_entity['value']
                normalized_location_text = _normalize_russian_text_with_pymorphy(user_specified_location)
                logger.debug(f"Timezone location entity: '{user_specified_location}', normalized: '{normalized_location_text}'")

                mapped_tz_db_name = None
                # Sort keys by length descending to match longer phrases first (e.g., "нью йорк" before "йорк")
                sorted_city_keys = sorted(_PRE_NORMALIZED_CITY_TO_TIMEZONE.keys(), key=len, reverse=True)

                for known_norm_city in sorted_city_keys:
                    tz_db_name = _PRE_NORMALIZED_CITY_TO_TIMEZONE[known_norm_city]
                    # Use regex for whole word matching. re.escape handles special characters in city names.
                    if re.search(r'\b' + re.escape(known_norm_city) + r'\b', normalized_location_text, re.IGNORECASE):
                        mapped_tz_db_name = tz_db_name
                        logger.info(f"Mapped NLU timezone location '{user_specified_location}' (normalized: '{normalized_location_text}') to city '{known_norm_city}' -> {mapped_tz_db_name} using regex.")
                        break
                
                # Fallback for exact match on normalized text if regex didn't catch it (e.g. single word city name)
                if not mapped_tz_db_name and normalized_location_text in _PRE_NORMALIZED_CITY_TO_TIMEZONE:
                    mapped_tz_db_name = _PRE_NORMALIZED_CITY_TO_TIMEZONE[normalized_location_text]
                    logger.info(f"Mapped NLU timezone location '{user_specified_location}' (normalized: '{normalized_location_text}') to city '{normalized_location_text}' -> {mapped_tz_db_name} using direct exact match.")

                if mapped_tz_db_name:
                    resolved_timezone_arg = mapped_tz_db_name
                else:
                    # If our mapping failed, pass the raw user-specified string.
                    # The get_current_time tool will try it and handle pytz.UnknownTimeZoneError.
                    logger.warning(f"Could not map NLU timezone location '{user_specified_location}' (normalized: '{normalized_location_text}') to a known TZ database name. Passing raw value '{user_specified_location}' to tool.")
                    resolved_timezone_arg = user_specified_location # Pass original, unnormalized
            
            parsed_command_data["timezone_str_for_tool"] = resolved_timezone_arg
            logger.info(f"Offline command parsed for get_time: {parsed_command_data}")
            return parsed_command_data

        elif intent_name in ["activate_device", "deactivate_device", "set_attribute"]:
            # --- Extract Entities for Device Control ---
            device_description_parts = [e['value'] for e in entities if e['entity'] == 'device_description']
            location_parts = [e['value'] for e in entities if e['entity'] == 'location']
            
            raw_device_search_parts = device_description_parts + location_parts
            normalized_device_search_parts = [_normalize_russian_text_with_pymorphy(part) for part in raw_device_search_parts]
            device_search_text = " ".join(filter(None, normalized_device_search_parts)).strip()

            pronoun_entity = next((e for e in entities if e['entity'] == 'pronoun_device'), None)
            attribute_name_entity = next((e for e in entities if e['entity'] == 'attribute_name'), None)
            
            attribute_value_entities = [e for e in entities if e['entity'] == 'attribute_value']
            raw_attribute_values = sorted(attribute_value_entities, key=lambda e: e['start'])
            attribute_value_raw = " ".join(e['value'] for e in raw_attribute_values).strip() if raw_attribute_values else None
            
            primary_attribute_value_entity = raw_attribute_values[0] if raw_attribute_values else None
            if not primary_attribute_value_entity: # Check standalone number/percentage
                numeric_entity = next((e for e in entities if e['entity'] in ['percentage', 'number']), None)
                if numeric_entity:
                    primary_attribute_value_entity = numeric_entity # For group/role check
                    if not attribute_value_raw: # If not already set by 'attribute_value'
                         attribute_value_raw = numeric_entity['value']


            logger.debug(f"Extracted Device Entities: device_search='{device_search_text}' (raw: {' '.join(raw_device_search_parts)}), "
                         f"pronoun='{pronoun_entity['value'] if pronoun_entity else None}', "
                         f"attr_name='{attribute_name_entity['value'] if attribute_name_entity else None}', "
                         f"attr_value_raw='{attribute_value_raw}'")

            # --- Resolve Device ---
            available_devices = mqtt_client.get_device_friendly_names()
            if not available_devices:
                 logger.error("MQTT device list unavailable. Cannot resolve device.")
                 return None

            resolved_device = None
            if device_search_text:
                matched_device = find_best_match(device_search_text, available_devices, score_cutoff=70)
                if matched_device:
                    resolved_device = matched_device
                    _last_mentioned_device = resolved_device
                    logger.info(f"Device resolved via NLU entities '{device_search_text}' to '{resolved_device}'")
            
            if not resolved_device and pronoun_entity:
                if _last_mentioned_device and _last_mentioned_device in available_devices:
                    resolved_device = _last_mentioned_device
                    logger.info(f"Device resolved via pronoun to context device '{resolved_device}'")
                else:
                    logger.warning(f"Pronoun used, but no valid context device ('{_last_mentioned_device}').")
            
            if not resolved_device and intent_name in ['set_attribute', 'activate_device', 'deactivate_device']:
                 if _last_mentioned_device and _last_mentioned_device in available_devices:
                    resolved_device = _last_mentioned_device
                    logger.info(f"No device entity/pronoun, using context device '{resolved_device}' for intent '{intent_name}'")


            if not resolved_device:
                logger.error(f"Failed to resolve target device for: '{text}'. Search: '{device_search_text}'.")
                return None
            parsed_command_data["device"] = resolved_device

            # --- Determine Attribute and Raw Value for Device Control ---
            target_attribute_std_name = None
            raw_value_for_tool = None

            if intent_name == "activate_device":
                target_attribute_std_name = "state"
                raw_value_for_tool = "ON"
            elif intent_name == "deactivate_device":
                target_attribute_std_name = "state"
                raw_value_for_tool = "OFF"
            elif intent_name == "set_attribute":
                if attribute_name_entity:
                    nlu_attr_val = attribute_name_entity['value']
                    norm_nlu_attr_val = normalize_text_key(_normalize_russian_text_with_pymorphy(nlu_attr_val))
                    target_attribute_std_name = _NLU_ATTR_TO_STANDARD.get(norm_nlu_attr_val)
                    if not target_attribute_std_name:
                        target_attribute_std_name = norm_nlu_attr_val 
                        logger.warning(f"NLU attribute '{nlu_attr_val}' (normalized: {norm_nlu_attr_val}) not in standard map. Using as is.")
                    else:
                        logger.debug(f"Attribute from NLU entity '{nlu_attr_val}' mapped to '{target_attribute_std_name}'")
                
                if not target_attribute_std_name and primary_attribute_value_entity:
                    logger.debug(f"No explicit attribute_name. Guessing from attribute_value: '{attribute_value_raw}' (entity: {primary_attribute_value_entity})")
                    entity_group = primary_attribute_value_entity.get('group')
                    
                    if entity_group == 'color_values': target_attribute_std_name = "color"
                    elif entity_group == 'temp_presets': target_attribute_std_name = "color_temp"
                    elif entity_group == 'relative_brightness_values': target_attribute_std_name = "brightness"
                    elif primary_attribute_value_entity.get('entity') == 'percentage': target_attribute_std_name = "brightness"
                    else:
                        if attribute_value_raw:
                            normalized_val = normalize_text_key(_normalize_russian_text_with_pymorphy(attribute_value_raw))
                            if normalized_val in COLOR_NAME_TO_XY: target_attribute_std_name = "color"
                            elif normalized_val in COLOR_TEMP_PRESETS: target_attribute_std_name = "color_temp"
                    if target_attribute_std_name:
                         logger.info(f"Guessed attribute '{target_attribute_std_name}' from attribute_value entity.")
                
                if not target_attribute_std_name:
                    logger.error(f"Failed to determine target attribute for 'set_attribute'. Text: '{text}'")
                    return None
                
                if attribute_value_raw is None:
                    logger.error(f"Intent 'set_attribute' for '{target_attribute_std_name}', but no attribute_value. Text: '{text}'")
                    return None
                
                if attribute_value_raw:
                    original_raw_value = attribute_value_raw
                    normalized_value_for_tool = _normalize_russian_text_with_pymorphy(original_raw_value)
                    if normalized_value_for_tool != original_raw_value:
                        logger.debug(f"Normalized attribute value '{original_raw_value}' to '{normalized_value_for_tool}' using Pymorphy3.")
                    raw_value_for_tool = normalized_value_for_tool
                else: # Should have been caught by "attribute_value_raw is None" check above
                    raw_value_for_tool = None 
            
            parsed_command_data["attribute"] = target_attribute_std_name
            parsed_command_data["raw_value"] = raw_value_for_tool

            if not all([parsed_command_data["device"], parsed_command_data["attribute"], parsed_command_data["raw_value"] is not None]):
                 logger.error(f"Device command parsing incomplete: {parsed_command_data}. Text: '{text}'")
                 return None

            logger.info(f"Offline device command parsed: {parsed_command_data}")
            return parsed_command_data
        else:
            logger.warning(f"Unhandled intent '{intent_name}'. No command generated. Text: '{text}'")
            return None

    except httpx.RequestError as e:
        logger.error(f"Rasa NLU connection error {RASA_NLU_URL}: {e}")
    except httpx.HTTPStatusError as e:
        logger.error(f"Rasa NLU server error {e.response.status_code}. Response: {e.response.text[:500]}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from Rasa NLU: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during Rasa NLU parsing: {e}", exc_info=True)
    return None


async def execute_offline_command(parsed_command: Dict[str, Any]) -> str:
    """
    Executes the command parsed by `parse_offline_command`.
    """
    intent = parsed_command.get("intent")

    if intent == "get_time":
        timezone_arg = parsed_command.get("timezone_str_for_tool")
        logger.info(f"Executing get_time command. Timezone argument: '{timezone_arg}'")
        try:
            # get_current_time is a Langchain @tool, making it a Runnable.
            # We invoke it with a dictionary matching its arguments.
            result_message = get_current_time.invoke({"timezone_str": timezone_arg})
            return result_message
        except Exception as e:
            logger.error(f"Error calling get_current_time tool: {e}", exc_info=True)
            return "Извините, произошла внутренняя ошибка при попытке получить время."

    elif intent in ["activate_device", "deactivate_device", "set_attribute"]:
        device_name = parsed_command.get("device")
        attribute_name = parsed_command.get("attribute")
        raw_value_from_nlu = parsed_command.get("raw_value")

        # This check should ideally be guaranteed by parse_offline_command for these intents
        if not device_name or not attribute_name or raw_value_from_nlu is None:
            logger.error(f"Incomplete device command for execution: {parsed_command}")
            return "Внутренняя ошибка: неполная информация для управления устройством."

        logger.info(f"Executing offline device command: Device='{device_name}', Attribute='{attribute_name}', RawValue='{raw_value_from_nlu}'")
        try:
            value_str = str(raw_value_from_nlu)
            
            # Calling the set_device_attribute tool function
            # Assuming set_device_attribute is a Langchain @tool or a regular function
            if hasattr(set_device_attribute, 'invoke'): # If it's a Langchain Runnable tool
                 result_message = set_device_attribute.invoke({
                    "user_description": device_name,
                    "attribute": attribute_name,
                    "value": value_str,
                    "guessed_friendly_name": device_name
                })
            elif callable(set_device_attribute): # If it's a direct callable (possibly decorated)
                 result_message = set_device_attribute(
                    user_description=device_name,
                    attribute=attribute_name,
                    value=value_str,
                    guessed_friendly_name=device_name
                )
            else:
                logger.error("set_device_attribute is not callable as expected.")
                return "Внутренняя ошибка: функция управления устройством настроена неверно."
            
            return result_message
        except Exception as e:
            logger.error(f"Error calling set_device_attribute for offline command: {e}", exc_info=True)
            return f"Извините, произошла внутренняя ошибка при попытке управления устройством '{device_name}' (атрибут: '{attribute_name}')."
    else:
        logger.warning(f"Execute_offline_command received unhandled intent: {intent}")
        return f"Извините, я не знаю, как обработать команду для намерения '{intent}'."