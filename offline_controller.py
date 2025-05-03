import logging
import json # Added for execute_offline_command and Rasa interaction
from typing import Dict, Optional, List, Any, Tuple

from thefuzz import process as fuzz_process, fuzz
import httpx # Added HTTP client for Rasa

from tools.tools import (
    COLOR_NAME_TO_XY,
    COLOR_TEMP_PRESETS,
    normalize_text_key,
    parse_value as base_parse_value, # Renamed original parse_value to avoid conflict
    find_capability
)
from mqtt_client import mqtt_client
from settings import settings

logger = logging.getLogger(__name__)

# --- State ---
_last_mentioned_device: Optional[str] = None

# --- Rasa NLU API Configuration ---
RASA_NLU_URL = settings.rasa_nlu.url # Get URL from settings
RASA_NLU_TIMEOUT = settings.rasa_nlu.timeout # Get timeout from settings
RASA_INTENT_CONFIDENCE_THRESHOLD = settings.rasa_nlu.intent_confidence_threshold # Get threshold from settings

# --- Helper Functions ---

def find_best_match(query: str, choices: List[str], scorer=fuzz.token_sort_ratio, score_cutoff=75) -> Optional[str]:
    """Finds the best fuzzy match above a cutoff score."""
    if not choices: # Added check for empty list
        return None
    match = fuzz_process.extractOne(query, choices, scorer=scorer, score_cutoff=score_cutoff)
    return match[0] if match else None

# This function is now primarily used by execute_offline_command
# to interpret the 'attribute_value' string extracted by Rasa.
def parse_value_offline(value_str: str, target_type: str, capability_details: Optional[Dict] = None) -> Tuple[Optional[Any], Optional[str]]:
    """
    Parses value string (likely from Rasa entity) for offline commands,
    handling presets, percentages etc. based on the target attribute type.
    Returns (parsed_value, representation_for_message)
    """
    if value_str is None: # Handle None input gracefully
        return None, None
    value_str = str(value_str).strip().lower() # Normalize to lower and strip
    normalized_value_key = normalize_text_key(value_str) # Further normalization if needed

    # 1. Color Names (Target type hints we expect a color)
    if target_type == "color_name":
        # Check exact normalized match first
        if normalized_value_key in COLOR_NAME_TO_XY:
            return normalized_value_key, normalized_value_key # Return the normalized key
        else:
            # Try fuzzy matching against known color names
            match = find_best_match(normalized_value_key, list(COLOR_NAME_TO_XY.keys()), score_cutoff=75)
            if match:
                 logger.info(f"Fuzzy matched color '{value_str}' to '{match}'")
                 return match, match # Return the matched key
            else:
                 logger.warning(f"Could not parse '{value_str}' as a known color name.")
                 return None, None

    # 2. Color Temp (Target type hints color_temp, capability_details might be needed for range, but not parsing presets)
    elif target_type == "numeric_or_preset" and capability_details and capability_details.get('name') == 'color_temp':
        # Try exact preset match
        preset_mired = COLOR_TEMP_PRESETS.get(normalized_value_key)
        if preset_mired is not None:
            return preset_mired, f"{normalized_value_key} ({preset_mired} Mired)"

        # Try fuzzy preset match
        match = find_best_match(normalized_value_key, list(COLOR_TEMP_PRESETS.keys()), score_cutoff=75)
        if match:
            preset_mired = COLOR_TEMP_PRESETS[match]
            logger.info(f"Fuzzy matched color temp '{value_str}' to preset '{match}' ({preset_mired} Mired)")
            return preset_mired, f"{match} ({preset_mired} Mired)"

        # Try parsing as a numeric Mired value (ignore Kelvin for now)
        if 'k' in value_str:
             logger.warning(f"Kelvin parsing ('{value_str}') not supported in offline parse_value. Ignoring.")
             return None, None
        # Use base_parse_value for numeric conversion
        parsed_numeric = base_parse_value(value_str, "numeric")
        if isinstance(parsed_numeric, (int, float)):
             # Return the numeric value directly
             return parsed_numeric, str(parsed_numeric)

        logger.warning(f"Could not parse '{value_str}' as color_temp preset or numeric Mired.")
        return None, None

    # 3. Brightness (Target type hints numeric, capability_details might be needed for range/percentage calc)
    elif target_type == "numeric" and capability_details and capability_details.get('name') == 'brightness':
        # Handle keywords extracted by Rasa (assuming Rasa extracts "половина", "максимум" etc. as attribute_value)
        if normalized_value_key in ["половина", "середина"]:
            return "50%", "50%" # Return percentage string, execute will handle conversion
        if normalized_value_key in ["максимум", "максимальная", "полная", "ярко", "ярче"]: # Added "ярче" for robustness
            return "100%", "100%"
        if normalized_value_key in ["минимум", "минимальная", "тускло", "тусклее"]: # Added "тусклее"
             return "1%", "1%" # Use 1% as minimum often works better than 0%

        # Handle explicit percentages (e.g., "50 процентов" -> Rasa extracts "50 процентов" or just "50"?)
        # Assume Rasa might extract "50" or "50 процентов". We check for '%' or 'процент'
        cleaned_value_str = value_str.replace('процентов', '').replace('процента', '').strip()
        if '%' in cleaned_value_str or (cleaned_value_str.isdigit() and ('%' in value_str or 'процент' in value_str)):
             try:
                 percent_str = cleaned_value_str.replace('%','').strip()
                 percent = float(percent_str)
                 percent = max(0.0, min(100.0, percent))
                 # Return the percentage string for execute_offline_command to handle
                 return f"{percent}%", f"{percent}%"
             except ValueError:
                  logger.warning(f"Could not parse brightness percentage from '{value_str}'.")
                  # Fall through to try numeric parsing

        # Try parsing as a plain number (absolute value)
        parsed_numeric = base_parse_value(value_str, "numeric")
        if isinstance(parsed_numeric, (int, float)):
            # Return the numeric value directly
            return parsed_numeric, str(parsed_numeric)

        logger.warning(f"Could not parse '{value_str}' as numeric brightness, percentage, or keyword.")
        return None, None

    # 4. State (Target type hints binary)
    elif target_type == "binary":
         # Use base_parse_value which handles ON/OFF/TOGGLE and synonyms
         parsed_state = base_parse_value(value_str, "binary")
         if parsed_state:
             return parsed_state, parsed_state
         else:
             logger.warning(f"Could not parse '{value_str}' as a binary state (ON/OFF/TOGGLE).")
             return None, None

    # 5. Enum (Target type hints enum)
    elif target_type == "enum":
         # For enums, we usually return the string value as is.
         # Validation against allowed values happens in execute_offline_command.
         # We can still normalize it.
         return normalized_value_key, value_str # Return normalized key and original string

    # 6. Fallback for other numeric or generic string types
    else:
        # Use the base parser for simple numeric or return string as is
        parsed = base_parse_value(value_str, target_type)
        if parsed is not None:
            return parsed, str(parsed)
        else:
            # If base parser fails for non-specific types, return original string
            logger.debug(f"Returning raw string '{value_str}' for target_type '{target_type}' after base_parse_value failed.")
            return value_str, value_str


async def parse_offline_command(text: str) -> Optional[Dict[str, Any]]:
    """
    Parses user text command by calling the Rasa NLU HTTP API.
    Extracts intent and entities, resolves the device using context,
    and prepares a command dictionary for execution.
    """
    global _last_mentioned_device
    payload = {"text": text}
    # Initialize with None, especially for value which might be 0
    parsed_command = {"device": None, "attribute": None, "value": None, "value_repr": None, "intent": None}
    logger.debug(f"Attempting offline parse via Rasa NLU API (URL: {RASA_NLU_URL}) for: '{text}'")

    if not RASA_NLU_URL:
        logger.error("Rasa NLU URL is not configured in settings. Cannot parse offline command.")
        return None

    try:
        async with httpx.AsyncClient(timeout=RASA_NLU_TIMEOUT) as client:
            response = await client.post(RASA_NLU_URL, json=payload)
            response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
            result = response.json()
        logger.debug(f"Rasa NLU API raw result: {json.dumps(result, indent=2, ensure_ascii=False)}")

        # --- Interpret Rasa Response ---
        intent_data = result.get("intent", {})
        intent = intent_data.get("name")
        confidence = intent_data.get("confidence", 0.0)
        entities = result.get("entities", [])
        parsed_command["intent"] = intent

        # Check intent confidence
        if not intent or confidence < RASA_INTENT_CONFIDENCE_THRESHOLD:
            logger.warning(f"NLU confidence low ({confidence:.2f} < {RASA_INTENT_CONFIDENCE_THRESHOLD}) for intent '{intent}'. Text: '{text}'. Aborting offline command.")
            return None # Not confident enough in the intent

        logger.info(f"Rasa NLU classified intent as '{intent}' with confidence {confidence:.2f}")

        # --- Extract Entities ---
        # Combine device description and location for searching
        device_parts = [e['value'] for e in entities if e['entity'] in ['device_description', 'location']]
        device_search_text = " ".join(device_parts).strip()
        # Find pronoun
        pronoun_entity = next((e for e in entities if e['entity'] == 'pronoun_device'), None)
        # Find attribute name
        attribute_name_entity = next((e for e in entities if e['entity'] == 'attribute_name'), None)
        # Find attribute value(s) - join if multiple parts (e.g., "холодный белый")
        attribute_value_entities = [e['value'] for e in entities if e['entity'] == 'attribute_value']
        attribute_value_raw = " ".join(attribute_value_entities).strip() if attribute_value_entities else None

        logger.debug(f"Extracted Entities: device_search='{device_search_text}', pronoun='{pronoun_entity['value'] if pronoun_entity else None}', attr_name='{attribute_name_entity['value'] if attribute_name_entity else None}', attr_value='{attribute_value_raw}'")

        # --- Resolve Device ---
        available_devices = mqtt_client.get_device_friendly_names()
        if not available_devices:
             # Log error but maybe proceed if context exists? Or fail? Let's fail.
             logger.error("MQTT device list is unavailable. Cannot resolve device for offline command.")
             return None

        resolved_device = None
        if device_search_text:
            logger.debug(f"Searching device for NLU parts: '{device_search_text}' in {available_devices}")
            # Lower score cutoff slightly for NLU results as they might be less precise
            matched_device = find_best_match(device_search_text, available_devices, score_cutoff=70)
            if matched_device:
                resolved_device = matched_device
                _last_mentioned_device = resolved_device # Update context
                logger.info(f"Device resolved via NLU entities '{device_search_text}' to '{resolved_device}'")
            else:
                 logger.warning(f"NLU extracted device parts '{device_search_text}', but no suitable match found in available devices.")
        elif pronoun_entity:
            if _last_mentioned_device and _last_mentioned_device in available_devices:
                resolved_device = _last_mentioned_device
                logger.info(f"Device resolved via pronoun '{pronoun_entity['value']}' to context device '{resolved_device}'")
            else:
                logger.warning(f"Pronoun '{pronoun_entity['value']}' used, but no valid context device ('{_last_mentioned_device}') available.")
        # Implicit context: If intent is set_attribute and no device/pronoun found, assume last mentioned device
        elif intent == 'set_attribute' and _last_mentioned_device and _last_mentioned_device in available_devices:
             resolved_device = _last_mentioned_device
             logger.info(f"No device entity/pronoun found for 'set_attribute', using context device '{resolved_device}'")
        # Implicit context for activate/deactivate might be too ambiguous, require explicit mention or pronoun
        elif intent in ['activate_device', 'deactivate_device'] and _last_mentioned_device and _last_mentioned_device in available_devices:
             # Let's allow context for on/off too, but log it clearly
             resolved_device = _last_mentioned_device
             logger.info(f"No device entity/pronoun found for '{intent}', using context device '{resolved_device}'")


        if not resolved_device:
            # If still no device, try a final fuzzy match on the *entire* input text? Risky.
            # Let's fail if no device could be resolved via entities, pronoun, or context.
            logger.error(f"Failed to resolve target device for text: '{text}'. Search text: '{device_search_text}', Pronoun: {pronoun_entity is not None}, Context: '{_last_mentioned_device}'.")
            # Maybe send a specific message back to user? For now, just fail parsing.
            return None
        parsed_command["device"] = resolved_device

        # --- Determine Attribute and Value based on Intent ---
        if intent == "activate_device":
            parsed_command["attribute"] = "state"
            parsed_command["value"] = "ON" # Use standard value, execute will map if needed
            parsed_command["value_repr"] = "ON"
        elif intent == "deactivate_device":
            parsed_command["attribute"] = "state"
            parsed_command["value"] = "OFF" # Use standard value
            parsed_command["value_repr"] = "OFF"
        elif intent == "set_attribute":
            target_attribute = None
            # 1. Use explicit attribute_name entity if present
            if attribute_name_entity:
                 # Normalize the extracted name to our standard internal names
                 norm_attr_from_nlu = normalize_text_key(attribute_name_entity['value'])
                 if norm_attr_from_nlu in ["яркость", "brightness"]: target_attribute = "brightness"
                 elif norm_attr_from_nlu in ["цвет", "color"]: target_attribute = "color"
                 elif norm_attr_from_nlu in ["температура", "temperature", "цветовая температура", "цвет температура"]: target_attribute = "color_temp"
                 # Add mappings for other potential attributes if your NLU extracts them
                 # elif norm_attr_from_nlu in ["режим", "mode"]: target_attribute = "mode" # Example
                 else: logger.warning(f"NLU extracted unknown attribute name: '{attribute_name_entity['value']}' (normalized: {norm_attr_from_nlu})")
                 logger.debug(f"Attribute determined from NLU entity '{attribute_name_entity['value']}': '{target_attribute}'")

            # 2. If no attribute name entity, try to guess based on value (less reliable)
            if not target_attribute and attribute_value_raw:
                 logger.debug(f"Attribute name entity missing, attempting to guess from value: '{attribute_value_raw}'")
                 # Check if value looks like a color name
                 color_val, _ = parse_value_offline(attribute_value_raw, "color_name")
                 if color_val:
                      target_attribute = "color"
                      logger.debug("Guessed attribute 'color' based on value.")
                 else:
                      # Check if value looks like a color temp preset or Mired
                      # Need fake capability details for parse_value_offline structure
                      temp_val, _ = parse_value_offline(attribute_value_raw, "numeric_or_preset", {"name": "color_temp"})
                      if temp_val is not None: # Check for None explicitly
                           target_attribute = "color_temp"
                           logger.debug("Guessed attribute 'color_temp' based on value.")
                      else:
                           # Check if value looks like brightness (number, %, keyword)
                           # Need fake capability details
                           bright_val, _ = parse_value_offline(attribute_value_raw, "numeric", {"name": "brightness"})
                           if bright_val is not None: # Check for None explicitly
                                target_attribute = "brightness"
                                logger.debug("Guessed attribute 'brightness' based on value.")
                           # else:
                               # Could add guesses for other types like enums if needed

            if not target_attribute:
                 logger.error(f"Failed to determine target attribute for intent 'set_attribute'. Attribute Entity: {attribute_name_entity}. Value: '{attribute_value_raw}'. Text: '{text}'")
                 return None # Cannot proceed without knowing which attribute to set

            parsed_command["attribute"] = target_attribute

            # Value: Use the raw extracted value string. Interpretation happens in execute_offline_command.
            if attribute_value_raw is None:
                 # Check if maybe the attribute *implies* the value (e.g., "сделай свет ярче" - NLU might miss 'ярче' as value)
                 # This is complex NLU logic. For now, require attribute_value for set_attribute.
                 logger.error(f"Intent is 'set_attribute' for attribute '{target_attribute}', but no attribute_value entity was extracted. Text: '{text}'")
                 return None
            parsed_command["value"] = attribute_value_raw
            parsed_command["value_repr"] = attribute_value_raw # Initial representation

        else:
            # Handle other intents like greet, goodbye? Or just ignore them for execution.
            logger.warning(f"Received unhandled intent '{intent}' from NLU. No command generated. Text: '{text}'")
            return None

        # --- Final Validation ---
        # Check if all essential parts are present (value can be 0, so check for None)
        if not all([parsed_command["device"], parsed_command["attribute"], parsed_command["value"] is not None]):
             logger.error(f"Offline command parsing incomplete after NLU processing. Result: {parsed_command}. Text: '{text}'")
             return None

        logger.info(f"Offline command successfully parsed via Rasa NLU API: {parsed_command}")
        return parsed_command

    except httpx.RequestError as e:
        logger.error(f"Could not connect to Rasa NLU server at {RASA_NLU_URL}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        # Log the response body if available for more detailed errors
        err_body = e.response.text
        logger.error(f"Rasa NLU server returned error {e.response.status_code}. Response: {err_body[:500]}") # Log first 500 chars
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from Rasa NLU server: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during offline command parsing with Rasa NLU: {e}", exc_info=True)
        return None


async def execute_offline_command(parsed_command: Dict[str, Any]) -> str:
    """
    Executes the command parsed by Rasa NLU via MQTT.
    Performs final validation and value interpretation using parse_value_offline
    based on device capabilities.
    """
    device_name = parsed_command.get("device")
    # Standardized attribute name ('state', 'brightness', 'color', 'color_temp')
    attribute_name = parsed_command.get("attribute")
    # Raw value string extracted by Rasa (e.g., "ON", "OFF", "синий", "50 процентов", "теплый", "75")
    raw_value_from_nlu = parsed_command.get("value")
    # Initial representation, might be refined after parsing
    value_repr = parsed_command.get("value_repr", str(raw_value_from_nlu))

    # --- Basic Checks ---
    if not device_name or not attribute_name or raw_value_from_nlu is None:
        logger.error(f"Incomplete command received for execution: {parsed_command}")
        return "Internal error: Incomplete command information."

    logger.info(f"Executing offline command: Device='{device_name}', Attribute='{attribute_name}', RawValue='{raw_value_from_nlu}'")

    # --- Get Device Capabilities ---
    capabilities = mqtt_client.get_device_capabilities(device_name)
    if capabilities is None:
        # Device might be offline or name is wrong despite NLU match
        logger.error(f"Could not get capabilities for device '{device_name}'. Cannot execute command.")
        return f"Sorry, I couldn't get the current status or capabilities for '{device_name}'. Is it online?"

    # --- Find Specific Capability Details ---
    # Use the *standardized* attribute_name from parsed_command
    capability_details = find_capability(capabilities, attribute_name)
    if not capability_details:
        available_attrs = [k for k, v in capabilities.items() if v.get("access", 0) & 2] # Writable attributes
        logger.warning(f"Standardized attribute '{attribute_name}' not found or not exposed in capabilities for '{device_name}'. Available writable: {available_attrs}")
        # Try fuzzy matching the standardized name against available ones? Less useful here.
        return f"Sorry, device '{device_name}' doesn't seem to support changing '{attribute_name}'. Writable attributes: {available_attrs}"

    # --- Check Access Rights ---
    if not (capability_details.get("access", 0) & 2): # Check if bit 1 (write) is set
        cap_name_display = capability_details.get('name', attribute_name)
        logger.warning(f"Attribute '{cap_name_display}' on '{device_name}' is read-only (access: {capability_details.get('access')}).")
        return f"Sorry, the attribute '{cap_name_display}' on '{device_name}' cannot be changed."

    # --- Final Value PARSING, Validation and Payload Creation ---
    cap_type = capability_details.get("type") # e.g., 'numeric', 'binary', 'enum', 'composite'
    cap_name_from_expose = capability_details.get('name', attribute_name) # Actual name in expose
    mqtt_property_name = capability_details.get('property', cap_name_from_expose) # MQTT key
    payload_dict = None
    final_value = None # The value after parsing and validation, ready for MQTT

    # Determine the expected type for parse_value_offline based on the *standardized* attribute name
    expected_parse_type = "string" # Default fallback
    if attribute_name == 'state': expected_parse_type = "binary"
    elif attribute_name == 'brightness': expected_parse_type = "numeric" # Will handle % within parse_value_offline
    elif attribute_name == 'color_temp': expected_parse_type = "numeric_or_preset" # Handles presets/Mired
    elif attribute_name == 'color': expected_parse_type = "color_name" # Expects color name
    elif cap_type: expected_parse_type = cap_type # Use type from expose if attribute is less standard

    logger.debug(f"Parsing raw NLU value '{raw_value_from_nlu}' using target_type '{expected_parse_type}' for attribute '{attribute_name}' (capability: {cap_name_from_expose})")
    # Use parse_value_offline to interpret the raw NLU string based on expected type
    parsed_value, parsed_repr = parse_value_offline(raw_value_from_nlu, expected_parse_type, capability_details)

    if parsed_value is None:
         # Parsing failed for the given type
         logger.error(f"Failed to parse NLU value '{raw_value_from_nlu}' for attribute '{cap_name_from_expose}' (expected type: {expected_parse_type})")
         # Provide more specific error based on type?
         if expected_parse_type == "color_name":
             return f"Sorry, I couldn't understand '{raw_value_from_nlu}' as a known color name."
         elif expected_parse_type == "numeric_or_preset" and attribute_name == "color_temp":
             presets = list(COLOR_TEMP_PRESETS.keys())
             return f"Sorry, '{raw_value_from_nlu}' isn't a known color temperature preset (like {presets[0]}, {presets[1]}...) or a valid Mired number."
         elif expected_parse_type == "numeric" and attribute_name == "brightness":
              return f"Sorry, I couldn't understand '{raw_value_from_nlu}' as a brightness level (number, percentage, or keyword like 'maximum')."
         else:
              return f"Sorry, I couldn't understand the value '{raw_value_from_nlu}' for {cap_name_from_expose}."

    # Use the potentially better representation from the parser
    if parsed_repr: value_repr = parsed_repr
    final_value = parsed_value # This is the value we'll validate and send
    logger.info(f"Parsed NLU value '{raw_value_from_nlu}' to: {final_value} (Type: {type(final_value)}), Representation: '{value_repr}'")

    # --- Payload Creation & Validation (using final_value) ---

    # a) Color (final_value should be a normalized color name string)
    if attribute_name == 'color':
        features = capability_details.get('features', [])
        supports_xy = any(f.get('name') == 'color_xy' for f in features)
        # Check if the main capability itself is 'color_xy' or if it's composite with xy feature
        is_composite_or_xy = cap_type == 'composite' or cap_name_from_expose == 'color_xy'

        if not (supports_xy or is_composite_or_xy):
             logger.warning(f"Device '{device_name}' capability '{cap_name_from_expose}' doesn't support XY color setting.")
             return f"Sorry, '{device_name}' doesn't seem to support setting color by name/XY coordinates."

        # final_value should be a string (color name) from parse_value_offline
        if isinstance(final_value, str) and final_value in COLOR_NAME_TO_XY:
            xy_coords = COLOR_NAME_TO_XY[final_value]
            # Payload structure depends on whether it's a composite 'color' object or direct 'color_xy'
            if cap_type == 'composite' and mqtt_property_name == 'color': # Common case Z2M
                 payload_dict = {mqtt_property_name: {"x": xy_coords[0], "y": xy_coords[1]}}
            elif cap_name_from_expose == 'color_xy': # Direct XY capability
                 payload_dict = {mqtt_property_name: {"x": xy_coords[0], "y": xy_coords[1]}}
            else: # Fallback? Maybe just send XY under the property name? Less likely.
                 logger.warning(f"Ambiguous color capability structure for '{device_name}'. Sending XY under '{mqtt_property_name}'.")
                 payload_dict = {mqtt_property_name: {"x": xy_coords[0], "y": xy_coords[1]}}

            logger.info(f"Converted color '{final_value}' to XY {xy_coords} for MQTT property '{mqtt_property_name}'")
        else:
             # Should not happen if parse_value_offline worked correctly
             logger.error(f"Internal error: Invalid final_value '{final_value}' (type: {type(final_value)}) for color after parsing.")
             return f"Internal error processing color '{value_repr}'."

    # b) Color Temp (final_value should be a numeric Mired value)
    elif attribute_name == 'color_temp':
        if isinstance(final_value, (int, float)):
            # Range check
            min_val = capability_details.get("value_min")
            max_val = capability_details.get("value_max")
            is_valid, range_str = True, ""
            if min_val is not None: range_str += f"min {min_val}"
            if max_val is not None: range_str += f"{' ' if range_str else ''}max {max_val}"
            if min_val is not None and final_value < min_val: is_valid = False
            if max_val is not None and final_value > max_val: is_valid = False

            if not is_valid:
                logger.warning(f"Value {final_value} ({value_repr}) for 'color_temp' is outside range ({range_str}) for '{device_name}'.")
                return f"Sorry, the value {final_value} ({value_repr}) for color temperature is outside the allowed range ({range_str}) for '{device_name}'."

            payload_dict = {mqtt_property_name: final_value}
        else:
             # Should not happen if parse_value_offline worked
             logger.error(f"Internal error: Invalid final_value type '{type(final_value)}' for color_temp after parsing.")
             return f"Internal error processing color temperature value '{value_repr}'."

    # c) Brightness (final_value can be numeric OR a string like "X%")
    elif attribute_name == 'brightness':
        numeric_brightness = None
        # Check if parse_value_offline returned a percentage string
        if isinstance(final_value, str) and '%' in final_value:
            try:
                percent = float(final_value.replace('%','').strip())
                # Get range for conversion
                min_val = capability_details.get("value_min")
                max_val = capability_details.get("value_max")
                # Use defaults if range not specified (common for brightness: 0-254 or 1-254)
                abs_max = float(max_val) if max_val is not None else 254.0
                abs_min = float(min_val) if min_val is not None else 0.0 # Use 0 as default min unless specified
                # Calculate absolute value based on percentage and device range
                calc_value = abs_min + (abs_max - abs_min) * (percent / 100.0)
                # Round to int unless unit suggests float (rare for brightness)
                numeric_brightness = calc_value if capability_details.get("unit") else int(round(calc_value))
                # Ensure calculated value is within absolute min/max (rounding might exceed)
                numeric_brightness = max(abs_min, min(abs_max, numeric_brightness))
                logger.info(f"Converted brightness {percent}% ({final_value}) to absolute value {numeric_brightness} (Device range: {abs_min}-{abs_max})")
            except (ValueError, TypeError) as e:
                 logger.error(f"Internal error converting brightness percentage '{final_value}': {e}")
                 return f"Internal error processing brightness percentage '{value_repr}'."
        elif isinstance(final_value, (int, float)):
            # Value was already parsed as numeric
            numeric_brightness = final_value
        else:
             logger.error(f"Internal error: Invalid final_value type '{type(final_value)}' for brightness after parsing.")
             return f"Internal error processing brightness value '{value_repr}'."

        # Range check (applies to both direct numeric and converted percentage)
        if numeric_brightness is not None:
            min_val = capability_details.get("value_min")
            max_val = capability_details.get("value_max")
            is_valid, range_str = True, ""
            if min_val is not None: range_str += f"min {min_val}"
            if max_val is not None: range_str += f"{' ' if range_str else ''}max {max_val}"
            # Use numeric_brightness for check
            if min_val is not None and numeric_brightness < min_val: is_valid = False
            if max_val is not None and numeric_brightness > max_val: is_valid = False

            if not is_valid:
                logger.warning(f"Value {numeric_brightness} ({value_repr}) for 'brightness' is outside range ({range_str}) for '{device_name}'.")
                return f"Sorry, the brightness value {numeric_brightness} ({value_repr}) is outside the allowed range ({range_str}) for '{device_name}'."

            payload_dict = {mqtt_property_name: numeric_brightness}
        # else: payload_dict remains None if conversion failed

    # d) State (final_value should be 'ON', 'OFF', or 'TOGGLE')
    elif attribute_name == 'state':
         # parse_value_offline should return standardized 'ON', 'OFF', 'TOGGLE'
         if final_value not in ["ON", "OFF", "TOGGLE"]:
              logger.error(f"Internal error: Invalid final_value '{final_value}' for state after parsing.")
              return f"Internal error processing state value '{value_repr}'."

         # Map standard ON/OFF/TOGGLE to device-specific values if defined in capabilities
         val_on = capability_details.get("value_on", "ON")
         val_off = capability_details.get("value_off", "OFF")
         val_toggle = capability_details.get("value_toggle") # Often not present

         mqtt_value = final_value # Start with the standard value
         if final_value == "ON": mqtt_value = val_on
         elif final_value == "OFF": mqtt_value = val_off
         elif final_value == "TOGGLE":
              # Only use value_toggle if it exists, otherwise send standard "TOGGLE"
              if val_toggle:
                   mqtt_value = val_toggle
              else:
                   # If device doesn't explicitly list value_toggle, sending "TOGGLE" might still work for some
                   mqtt_value = "TOGGLE"
                   logger.debug(f"Sending standard 'TOGGLE' for {device_name} as value_toggle is not defined.")

         payload_dict = {mqtt_property_name: mqtt_value}
         value_repr = str(mqtt_value) # Update representation to what's actually sent

    # e) Other types (Enum, Numeric - if not brightness/temp)
    else:
        # Handle generic numeric and enum based on cap_type
        if cap_type == "numeric":
             if not isinstance(final_value, (int, float)):
                  logger.error(f"Expected numeric type for '{cap_name_from_expose}', but got {type(final_value)} after parsing '{raw_value_from_nlu}'.")
                  return f"Sorry, expected a number for '{cap_name_from_expose}', got '{value_repr}'."
             # Range check
             min_val = capability_details.get("value_min")
             max_val = capability_details.get("value_max")
             is_valid, range_str = True, ""
             if min_val is not None: range_str += f"min {min_val}"
             if max_val is not None: range_str += f"{' ' if range_str else ''}max {max_val}"
             if min_val is not None and final_value < min_val: is_valid = False
             if max_val is not None and final_value > max_val: is_valid = False
             if not is_valid:
                  logger.warning(f"Value {final_value} ({value_repr}) for '{cap_name_from_expose}' is outside range ({range_str}).")
                  return f"Sorry, the value {final_value} ({value_repr}) is outside the allowed range ({range_str})."
             payload_dict = {mqtt_property_name: final_value}

        elif cap_type == "enum":
             allowed_enum = capability_details.get("values")
             if not allowed_enum:
                  logger.error(f"Enum capability '{cap_name_from_expose}' for '{device_name}' has no defined values list.")
                  return f"Sorry, allowed options for '{cap_name_from_expose}' are not defined for this device."

             # final_value should be the string from parse_value_offline
             if not isinstance(final_value, str):
                  logger.error(f"Expected string for enum '{cap_name_from_expose}', but got {type(final_value)} after parsing '{raw_value_from_nlu}'.")
                  return f"Internal error processing option '{value_repr}' for '{cap_name_from_expose}'."

             # Check if the parsed value is in the allowed list
             if final_value not in allowed_enum:
                  # Try fuzzy matching the *parsed* value against the allowed list
                  match = find_best_match(final_value, allowed_enum, score_cutoff=80)
                  if match:
                       logger.info(f"Fuzzy matched provided enum value '{final_value}' to allowed value '{match}'. Using match.")
                       final_value = match # Use the matched value
                  else:
                       logger.warning(f"Invalid enum value '{final_value}' ({value_repr}) for '{cap_name_from_expose}'. Allowed: {allowed_enum}")
                       return f"Sorry, '{value_repr}' is not a valid option for '{cap_name_from_expose}'. Allowed options: {', '.join(allowed_enum)}"

             payload_dict = {mqtt_property_name: final_value}
             value_repr = str(final_value) # Update repr to the final validated value

        # Fallback for 'string' or other unhandled types
        else:
             logger.warning(f"Executing command for generic/unhandled capability type '{cap_type}' for '{cap_name_from_expose}'. Sending parsed value '{final_value}' as is.")
             # Assume final_value is suitable for sending directly
             payload_dict = {mqtt_property_name: final_value}
             # value_repr should be okay from parsing step


    # --- Send Command ---
    if payload_dict is None:
        # This should only happen if there was an error during value processing/validation
        logger.error(f"Payload dictionary is None after validation. Command: {parsed_command}, ParsedValue: {final_value}, Attribute: {attribute_name}")
        # Return a generic error as specific issues should have been caught earlier
        return f"Internal error: Could not prepare the command for '{cap_name_from_expose}' with value '{value_repr}'."

    try:
        payload_json = json.dumps(payload_dict)
    except TypeError as e:
        logger.error(f"Failed to serialize payload dictionary to JSON: {payload_dict}. Error: {e}")
        return f"Internal error: Could not format the command for MQTT."

    topic = f"{settings.mqtt_broker.default_topic_base}/{device_name}/set"

    logger.info(f"Attempting OFFLINE set via Rasa: Device='{device_name}', Topic='{topic}', Payload='{payload_json}'")

    # Schedule the publish operation
    success = mqtt_client.schedule_publish(topic, payload_json, qos=1)

    if success:
        # Use the refined value_repr for the confirmation message
        return f"Okay, done. '{device_name}' {cap_name_from_expose} set to {value_repr}."
    else:
        # schedule_publish currently always returns True, but check anyway
        logger.error(f"MQTT client failed to schedule publish for topic '{topic}'.")
        return f"Sorry, there was an issue sending the offline command for '{device_name}'."