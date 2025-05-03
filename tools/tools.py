from typing import List, Dict, Optional, Any
import json
import logging
from thefuzz import process as fuzz_process, fuzz

from langchain_core.tools import tool
from mqtt_client import mqtt_client
from settings import settings
from .memory import manage_mem, search_mem

logger = logging.getLogger(__name__)

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

# --- End Preset Handling ---


@tool
def say_to_console(text: str) -> str:
    """Outputs a text to the console"""
    logger.info(f"[Tool Output] Console: {text}")
    return f"Okay, printed '{text}' to console."

def normalize_attribute_name(attr_name: str) -> str:
    """Приводит имя атрибута к нижнему регистру и заменяет пробелы на подчеркивания."""
    return attr_name.lower().replace(" ", "_")


def parse_value(value_str: str, target_type: str) -> Optional[Any]:
    """Пытается преобразовать строку значения в нужный тип (int, float, bool, str). Excludes preset logic."""
    try:
        if target_type == "numeric":
             # Пытаемся сначала в int, потом в float
             try: return int(value_str)
             except ValueError: return float(value_str)
        elif target_type == "binary":
            # Приводим к стандартным ON/OFF для простоты
            val_lower = value_str.strip().lower()
            # Расширяем синонимы
            if val_lower in ["on", "вкл", "включить", "включи", "1", "true", "да"]: return "ON"
            if val_lower in ["off", "выкл", "выключить", "выключи", "0", "false", "нет"]: return "OFF"
            if val_lower in ["toggle", "переключить", "переключи"]: return "TOGGLE"
            return None # Не распознано
        elif target_type == "enum":
            return value_str # Возвращаем как есть, проверка будет по списку
        else: # По умолчанию считаем строкой
            return value_str
    except (ValueError, TypeError):
        logger.warning(f"Could not parse value '{value_str}' as {target_type}")
        return None


def find_capability(capabilities: Dict[str, Dict[str, Any]], requested_attr: str) -> Optional[Dict[str, Any]]:
    """Ищет возможность по имени (с учетом нормализации) в словаре exposes."""
    if not capabilities: return None
    normalized_requested = normalize_attribute_name(requested_attr)

    # Пробуем прямое совпадение нормализованных имен
    for cap_name, details in capabilities.items():
        if normalize_attribute_name(cap_name) == normalized_requested:
            return details
        # Иногда имя может быть в 'name' внутри деталей
        if normalize_attribute_name(details.get("name", "")) == normalized_requested:
            return details

    logger.debug(f"Capability '{requested_attr}' (normalized: {normalized_requested}) not found directly in device capabilities.")
    return None


@tool
def set_device_attribute(
    user_description: str,
    attribute: str,
    value: str,
    guessed_friendly_name: Optional[str] = None
) -> str:
    """
    Sets a specific attribute of a Zigbee device based on its description.

    Use this to control features like on/off state, brightness, color temperature (by Mired value or preset like 'cool', 'warm', 'нейтральный'), or color by name.

    Parameters:
        user_description (str): The user's description of the device (e.g., 'light in the hall', 'kitchen lamp', 'свет в зале').
        attribute (str): The name of the attribute/capability to change (e.g., 'state', 'brightness', 'color_temp', 'color').
        value (str): The desired value. Examples: 'ON', 'OFF', '50', '128', 'blue', 'красный', '300', 'coolest', 'теплый', '4000K' (Kelvin currently not supported), 'warm white'. Provide the value as a string.
        guessed_friendly_name (Optional[str]): LLM's best guess for the exact 'friendly_name'. Leave empty or null if unsure.

    Example calls:
    - set_device_attribute(user_description='hall light', attribute='state', value='ON', guessed_friendly_name='hall_ceiling_1')
    - set_device_attribute(user_description='kitchen lamp', attribute='brightness', value='75')
    - set_device_attribute(user_description='свет в зале', attribute='state', value='OFF')
    - set_device_attribute(user_description='bedroom light', attribute='color_temp', value='350') # Mired value
    - set_device_attribute(user_description='study lamp', attribute='color_temp', value='cool') # Preset name
    - set_device_attribute(user_description='лампа-зал', attribute='color_temp', value='теплый') # Russian preset name
    - set_device_attribute(user_description='living room strip', attribute='color', value='red')
    - set_device_attribute(user_description='лампа-зал', attribute='color', value='синий')
    - set_device_attribute(user_description='office switch', attribute='state', value='TOGGLE')
    """
    if not user_description or not attribute or value is None:
        return "Error: Missing required parameters: user_description, attribute, or value."

    available_devices = mqtt_client.get_device_friendly_names()
    if not available_devices:
        return "Sorry, the device list is unavailable. Cannot control devices now."

    target_device_name = None
    match_method = "None"
    best_score = 0

    # 1. Найти целевое устройство (Logic remains the same)
    if guessed_friendly_name:
        guess_validation_threshold = 80
        exactish_match = None; highest_score = 0
        for device in available_devices:
            if normalize_attribute_name(guessed_friendly_name) == normalize_attribute_name(device):
                exactish_match = device
                highest_score = 100
                break
            score = fuzz.token_sort_ratio(guessed_friendly_name, device)
            if score >= guess_validation_threshold and score > highest_score:
                highest_score = score
                exactish_match = device

        if exactish_match:
            target_device_name = exactish_match
            best_score = highest_score
            match_method = f"LLM guess validated ('{target_device_name}' score: {best_score})"
        else:
            logger.warning(f"LLM guess '{guessed_friendly_name}' not validated (threshold: {guess_validation_threshold}).")

    if not target_device_name:
        description_match_threshold = 60
        logger.debug(f"Attempting fuzzy match for '{user_description}' against {len(available_devices)} devices using token_sort_ratio (threshold: {description_match_threshold})...")
        best_match_overall = fuzz_process.extractOne(
            user_description,
            available_devices,
            scorer=fuzz.token_sort_ratio
        )
        if best_match_overall:
            logger.info(f"Best fuzzy match for '{user_description}' is '{best_match_overall[0]}' with score {best_match_overall[1]} (using token_sort_ratio).")
            if best_match_overall[1] >= description_match_threshold:
                target_device_name = best_match_overall[0]
                best_score = best_match_overall[1]
                match_method = f"User description matched ('{target_device_name}' score: {best_score})"
            else:
                 logger.warning(f"Best match '{best_match_overall[0]}' score {best_match_overall[1]} is below threshold {description_match_threshold}.")

        if not target_device_name:
            error_msg = f"Sorry, I couldn't find a device matching '{user_description}' with sufficient confidence."
            if best_match_overall:
                error_msg += f" The closest match was '{best_match_overall[0]}' (score: {best_match_overall[1]}), but the threshold is {description_match_threshold}."
            else:
                 error_msg += " No potential matches found."
            error_msg += f" Available devices: {available_devices}"
            return error_msg

    logger.info(f"Target device identified: '{target_device_name}' (via {match_method}, score: {best_score})")

    # 2. Получить возможности устройства
    capabilities = mqtt_client.get_device_capabilities(target_device_name)
    if capabilities is None:
        return f"Sorry, could not retrieve capabilities for device '{target_device_name}'. It might be offline or recently removed."

    # 3. Найти запрошенную возможность (capability/attribute)
    capability_details = find_capability(capabilities, attribute)
    if not capability_details:
        available_attrs = list(capabilities.keys())
        logger.warning(f"Attribute '{attribute}' not found for device '{target_device_name}'. Available: {available_attrs}")
        fuzzy_attr_match = fuzz_process.extractOne(attribute, available_attrs, scorer=fuzz.token_sort_ratio, score_cutoff=80)
        if fuzzy_attr_match:
             suggestion = fuzzy_attr_match[0]
             return f"Sorry, device '{target_device_name}' doesn't directly support '{attribute}'. Did you mean '{suggestion}'? Available attributes: {available_attrs}"
        else:
             return f"Sorry, device '{target_device_name}' does not support the attribute '{attribute}'. Available attributes: {available_attrs}"

    # 4. Проверить права доступа
    if not (capability_details.get("access", 0) & 2):
        return f"Sorry, the attribute '{capability_details.get('name', attribute)}' on device '{target_device_name}' cannot be changed (read-only)."

    # --- 5. Преобразовать, валидировать значение и СОЗДАТЬ PAYLOAD ---
    cap_type = capability_details.get("type")
    cap_name_from_expose = capability_details.get('name', attribute)
    mqtt_attribute_name = capability_details.get('property', cap_name_from_expose)
    if not mqtt_attribute_name:
         logger.error(f"Could not determine the MQTT property name for capability: {capability_details}")
         return f"Internal error: Could not find the property name to control '{attribute}' on '{target_device_name}'."

    payload_dict = None
    final_value_repr = value # Start with raw value for messages

    # --- Handle 'color' ---
    if cap_name_from_expose == 'color':
        features = capability_details.get('features', [])
        supports_xy = any(f.get('name') == 'color_xy' for f in features)
        is_composite = cap_type == 'composite'

        if not (supports_xy or is_composite):
             return f"Sorry, the device '{target_device_name}' does not seem to support setting color via XY coordinates (required for color names)."

        normalized_color = normalize_text_key(value)
        if normalized_color in COLOR_NAME_TO_XY:
            xy_coords = COLOR_NAME_TO_XY[normalized_color]
            payload_dict = {"color": {"x": xy_coords[0], "y": xy_coords[1]}}
            final_value_repr = f"{normalized_color} (xy: {xy_coords})"
            logger.info(f"Mapped color name '{value}' to xy: {xy_coords} for device '{target_device_name}'")
        else:
            known_colors = list(COLOR_NAME_TO_XY.keys())
            color_match = fuzz_process.extractOne(normalized_color, known_colors, scorer=fuzz.token_sort_ratio, score_cutoff=75)
            if color_match:
                suggestion = color_match[0]
                return (f"Sorry, I don't recognize the color '{value}'. Did you mean '{suggestion}'? "
                        f"Known colors include: red, blue, green, yellow, purple, cyan, orange, pink...")
            else:
                return (f"Sorry, I don't recognize the color '{value}'. Please use common color names like "
                        f"red, blue, green, yellow, purple, cyan, orange, pink, or their Russian equivalents.")

    # --- Handle 'color_temp' (with presets) ---
    elif cap_name_from_expose == 'color_temp':
        normalized_value_key = normalize_text_key(value)
        preset_mired_value = COLOR_TEMP_PRESETS.get(normalized_value_key)
        final_mired_value = None # Store the final numeric Mired value

        if preset_mired_value is not None:
            # Found a matching preset
            logger.info(f"Mapped color temp preset '{normalized_value_key}' to Mired value {preset_mired_value}")
            final_mired_value = preset_mired_value
            # Use the original user value (or normalized key) in the response for clarity
            final_value_repr = f"{normalized_value_key} ({preset_mired_value} Mired)"
        else:
            # Not a preset, try parsing as a numeric Mired value
            # TODO: Add Kelvin support? Requires 1,000,000 / K conversion. Check if 'K' is in value.
            if 'k' in value.lower():
                 return f"Sorry, setting color temperature by Kelvin (e.g., '4000K') is not yet supported. Please use Mired values (e.g., 153-500) or presets like 'cool', 'warm'."

            parsed_as_numeric = parse_value(value, "numeric")
            if isinstance(parsed_as_numeric, (int, float)):
                final_mired_value = parsed_as_numeric
                final_value_repr = str(final_mired_value) # Use the number itself
            else:
                # Failed both preset lookup and numeric parsing
                known_presets_list = list(COLOR_TEMP_PRESETS.keys())
                # Try fuzzy match on presets?
                preset_match = fuzz_process.extractOne(normalized_value_key, known_presets_list, scorer=fuzz.token_sort_ratio, score_cutoff=75)
                if preset_match:
                    suggestion = preset_match[0]
                    return (f"Sorry, '{value}' is not a recognized color temperature preset or Mired value. "
                            f"Did you mean the preset '{suggestion}'?")
                else:
                    return (f"Sorry, '{value}' is not a recognized color temperature preset "
                            f"(like {', '.join(known_presets_list[:4])}...) "
                            f"or a valid numeric Mired value (typically 153-500).")

        # Proceed with range check using the determined final_mired_value
        if final_mired_value is not None:
            min_val = capability_details.get("value_min")
            max_val = capability_details.get("value_max")
            is_valid = True
            range_str = ""
            # Build range string for error message
            if min_val is not None: range_str += f"min {min_val}"
            if max_val is not None: range_str += f"{' ' if range_str else ''}max {max_val}"

            # Check validity
            if min_val is not None and final_mired_value < min_val: is_valid = False
            if max_val is not None and final_mired_value > max_val: is_valid = False

            if not is_valid:
                # Use the original user input 'value' in the error message for context
                return f"Sorry, the value {final_mired_value} (from '{value}') for 'color_temp' is outside the allowed range ({range_str}) for device '{target_device_name}'."
            else:
                # Value is valid, create payload
                payload_dict = {mqtt_attribute_name: final_mired_value}
                # final_value_repr is already set correctly above

    # --- Handle other types (numeric, binary, enum) ---
    else:
        # Use the generic parse_value based on capability type
        parsed_value = parse_value(value, cap_type)
        final_value = parsed_value # Keep track of the potentially modified value

        if final_value is None and cap_type != "enum":
             return f"Sorry, I couldn't understand the value '{value}' for the '{cap_name_from_expose}' attribute (expected type: {cap_type})."

        # Validation and potential modifications based on type
        if cap_type == "numeric":
            if not isinstance(final_value, (int, float)):
                 return f"Sorry, expected a numeric value for '{cap_name_from_expose}', but received '{value}'."

            min_val = capability_details.get("value_min")
            max_val = capability_details.get("value_max")

            # Handle brightness percentages BEFORE range check
            if cap_name_from_expose == 'brightness' and isinstance(value, str) and '%' in value:
                 try:
                     percent = float(value.replace('%','').strip())
                     percent = max(0.0, min(100.0, percent))
                     abs_max = float(max_val) if max_val is not None else 254.0
                     abs_min = float(min_val) if min_val is not None else 0.0
                     calc_value = abs_min + (abs_max - abs_min) * (percent / 100.0)
                     final_value = calc_value if capability_details.get("unit") else int(round(calc_value))
                     logger.info(f"Converted brightness {percent}% to absolute value {final_value} (range {abs_min}-{abs_max})")
                     # Update final_value_repr as well
                     final_value_repr = f"{percent}% ({final_value})"
                 except (ValueError, TypeError):
                     return f"Sorry, couldn't parse percentage value '{value}' for brightness."

            # Range check (using potentially updated final_value)
            is_valid = True
            range_str = ""
            if min_val is not None: range_str += f"min {min_val}"
            if max_val is not None: range_str += f"{' ' if range_str else ''}max {max_val}"
            if min_val is not None and final_value < min_val: is_valid = False
            if max_val is not None and final_value > max_val: is_valid = False

            if not is_valid:
                 return f"Sorry, the value {final_value} (from '{value}') for '{cap_name_from_expose}' is outside the allowed range ({range_str})."

            payload_dict = {mqtt_attribute_name: final_value}
            # Update final_value_repr if not already set by percentage logic
            if '%' not in value: final_value_repr = str(final_value)


        elif cap_type == "binary":
            allowed_vals = []
            val_on = capability_details.get("value_on", "ON")
            val_off = capability_details.get("value_off", "OFF")
            val_toggle = capability_details.get("value_toggle")
            allowed_vals.extend([val_on, val_off])
            if val_toggle: allowed_vals.append(val_toggle)

            if final_value not in allowed_vals:
                display_value = value if final_value is None else final_value
                return f"Sorry, invalid state '{display_value}' for '{cap_name_from_expose}'. Allowed values are usually: {allowed_vals}"
            else:
                payload_dict = {mqtt_attribute_name: final_value}
                final_value_repr = str(final_value)

        elif cap_type == "enum":
            allowed_enum = capability_details.get("values")
            if not allowed_enum:
                 logger.warning(f"Enum capability '{cap_name_from_expose}' for device '{target_device_name}' has no defined values!")
                 return f"Sorry, the allowed options for '{cap_name_from_expose}' are not defined for device '{target_device_name}'."

            if final_value not in allowed_enum:
                enum_match = fuzz_process.extractOne(final_value, allowed_enum, scorer=fuzz.token_sort_ratio, score_cutoff=75)
                if enum_match:
                     suggestion = enum_match[0]
                     return f"Sorry, invalid option '{final_value}' for '{cap_name_from_expose}'. Did you mean '{suggestion}'? Allowed options: {allowed_enum}"
                else:
                     return f"Sorry, invalid option '{final_value}' for '{cap_name_from_expose}'. Allowed options: {allowed_enum}"
            else:
                payload_dict = {mqtt_attribute_name: final_value}
                final_value_repr = str(final_value)

        else:
            logger.warning(f"Attribute '{cap_name_from_expose}' with unhandled type '{cap_type}' for device '{target_device_name}'. Value: '{value}'")
            # Try sending the raw value as a last resort if it's just 'string' type?
            if cap_type == "string":
                 payload_dict = {mqtt_attribute_name: value}
                 final_value_repr = value
                 logger.info(f"Treating attribute '{cap_name_from_expose}' as simple string. Sending raw value.")
            else:
                 return f"Sorry, I don't have specific instructions for setting the attribute '{cap_name_from_expose}' (type: {cap_type})."


    # --- 6. Send Command if payload was generated ---
    if payload_dict is None:
        logger.error(f"Payload dictionary was not generated for attribute '{cap_name_from_expose}' with value '{value}' on device '{target_device_name}'. This indicates a logic error.")
        return f"Internal error: Could not process the request to set '{cap_name_from_expose}' to '{value}' for '{target_device_name}'."

    payload_json = json.dumps(payload_dict)
    topic = f"{settings.mqtt_broker.default_topic_base}/{target_device_name}/set"

    logger.info(f"Attempting to set attribute '{mqtt_attribute_name}' for device '{target_device_name}'. Topic: {topic}, Payload: {payload_json}")

    success = mqtt_client.schedule_publish(topic, payload_json, qos=1)

    if success:
        # Use the potentially more descriptive final_value_repr
        return f"Okay, done. '{target_device_name}' {cap_name_from_expose} set to {final_value_repr}."
    else:
        return f"Sorry, there was an issue sending the command to set '{cap_name_from_expose}' for '{target_device_name}'."


TOOLS: List = [
    set_device_attribute,
    say_to_console,
    manage_mem,
    search_mem
]