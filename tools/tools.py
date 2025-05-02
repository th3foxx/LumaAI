from typing import List, Dict, Optional, Any
import json
import logging
from thefuzz import process as fuzz_process, fuzz # Добавили fuzz для прямого сравнения

from langchain_core.tools import tool
from mqtt_client import mqtt_client # Import the global instance
from settings import settings # Import settings for topic base
from .memory import manage_mem, search_mem

logger = logging.getLogger(__name__)


@tool
def say_to_console(text: str) -> str:
    """Outputs a text to the console"""
    logger.info(f"[Tool Output] Console: {text}")
    return f"Okay, printed '{text}' to console."

def normalize_attribute_name(attr_name: str) -> str:
    """Приводит имя атрибута к нижнему регистру и заменяет пробелы на подчеркивания."""
    return attr_name.lower().replace(" ", "_")


def parse_value(value_str: str, target_type: str) -> Optional[Any]:
    """Пытается преобразовать строку значения в нужный тип (int, float, bool, str)."""
    try:
        if target_type == "numeric":
             # Пытаемся сначала в int, потом в float
             try: return int(value_str)
             except ValueError: return float(value_str)
        elif target_type == "binary":
            # Приводим к стандартным ON/OFF для простоты
            val_lower = value_str.strip().lower()
            if val_lower in ["on", "вкл", "включить", "1", "true"]: return "ON"
            if val_lower in ["off", "выкл", "выключить", "0", "false"]: return "OFF"
            if val_lower in ["toggle", "переключить"]: return "TOGGLE"
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

    # Если не нашли, можно добавить нечеткий поиск по ключам capabilities
    # match = fuzz_process.extractOne(normalized_requested, capabilities.keys(), score_cutoff=85)
    # if match: return capabilities.get(match[0])

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

    Use this to control various features like on/off state, brightness, color temperature, etc.

    Parameters:
        user_description (str): The user's description of the device (e.g., 'light in the hall', 'kitchen lamp').
        attribute (str): The name of the attribute/capability to change (e.g., 'state', 'brightness', 'color_temp', 'color'). Try to use standard names found in Zigbee2MQTT exposes.
        value (str): The desired value for the attribute (e.g., 'ON', 'OFF', '50', '128', 'blue', '3000K', '400'). Provide the value as a string.
        guessed_friendly_name (Optional[str]): LLM's best guess for the exact 'friendly_name'. Leave empty or null if unsure.

    Example calls:
    - set_device_attribute(user_description='hall light', attribute='state', value='ON', guessed_friendly_name='hall_ceiling_1')
    - set_device_attribute(user_description='kitchen lamp', attribute='brightness', value='75') # Tool will try to handle percentage or absolute
    - set_device_attribute(user_description='bedroom light', attribute='color_temp', value='350') # Value in Mireds typically
    - set_device_attribute(user_description='living room strip', attribute='color', value='red') # Tool will try to convert color name
    - set_device_attribute(user_description='office switch', attribute='state', value='TOGGLE')
    """
    if not user_description or not attribute or value is None: # Проверяем value тоже
        return "Error: Missing required parameters: user_description, attribute, or value."

    available_devices = mqtt_client.get_device_friendly_names()
    if not available_devices:
        return "Sorry, the device list is unavailable. Cannot control devices now."

    target_device_name = None
    match_method = "None"

    # 1. Найти целевое устройство (логика из предыдущей версии)
    if guessed_friendly_name:
        guess_validation_threshold = 90
        exactish_match = None; highest_score = 0
        for device in available_devices:
            score = fuzz.token_sort_ratio(guessed_friendly_name, device)
            if score >= guess_validation_threshold and score > highest_score:
                highest_score = score; exactish_match = device
        if exactish_match:
            target_device_name = exactish_match
            match_method = f"LLM guess validated (score: {highest_score})"
        else: logger.warning(f"LLM guess '{guessed_friendly_name}' not validated.")

    if not target_device_name:
        description_match_threshold = 75
        best_match_tuple = fuzz_process.extractOne(user_description, available_devices, score_cutoff=description_match_threshold)
        if best_match_tuple:
            target_device_name = best_match_tuple[0]
            match_method = f"User description matched (score: {best_match_tuple[1]})"
        else:
            return f"Sorry, I couldn't find a device matching '{user_description}'."

    if not target_device_name: # Дополнительная проверка
         return f"Sorry, failed to identify the target device for '{user_description}'."

    logger.info(f"Target device identified: '{target_device_name}' (via {match_method})")

    # 2. Получить возможности устройства
    capabilities = mqtt_client.get_device_capabilities(target_device_name)
    if capabilities is None: # Может быть None, если устройство пропало между шагами
        return f"Sorry, could not retrieve capabilities for device '{target_device_name}'. It might be offline."

    # 3. Найти запрошенную возможность (capability/attribute)
    capability_details = find_capability(capabilities, attribute)
    if not capability_details:
        available_attrs = list(capabilities.keys())
        logger.warning(f"Attribute '{attribute}' not found for device '{target_device_name}'. Available: {available_attrs}")
        # Попробуем найти похожий атрибут
        fuzzy_attr_match = fuzz_process.extractOne(attribute, available_attrs, score_cutoff=80)
        if fuzzy_attr_match:
             suggestion = fuzzy_attr_match[0]
             return f"Sorry, device '{target_device_name}' doesn't directly support '{attribute}'. Did you mean '{suggestion}'? Available attributes: {available_attrs}"
        else:
             return f"Sorry, device '{target_device_name}' does not support the attribute '{attribute}'. Available attributes: {available_attrs}"

    # 4. Проверить права доступа (можно ли изменять атрибут)
    # access - битовая маска: 1=get, 2=set, 4=publish. Нам нужен бит 'set' (2).
    if not (capability_details.get("access", 0) & 2):
        return f"Sorry, the attribute '{capability_details['name']}' on device '{target_device_name}' cannot be changed (read-only)."

    # 5. Преобразовать и валидировать значение
    cap_type = capability_details.get("type")
    parsed_value = parse_value(value, cap_type)
    final_value = parsed_value # Значение для отправки в MQTT

    if parsed_value is None and cap_type != "enum": # Для enum None недопустим, если parse_value вернул строку
         return f"Sorry, I couldn't understand the value '{value}' for the '{capability_details['name']}' attribute."

    # Валидация для разных типов
    if cap_type == "numeric":
        min_val = capability_details.get("value_min")
        max_val = capability_details.get("value_max")
        # Проверяем диапазон, если он задан
        is_valid = True
        if min_val is not None and parsed_value < min_val: is_valid = False
        if max_val is not None and parsed_value > max_val: is_valid = False
        if not is_valid:
             return f"Sorry, the value {parsed_value} for '{capability_details['name']}' is outside the allowed range ({min_val}-{max_val})."
        # Обработка яркости в процентах (если атрибут 'brightness')
        if capability_details['name'] == 'brightness' and isinstance(value, str) and '%' in value:
             try:
                 percent = float(value.replace('%','').strip())
                 percent = max(0, min(100, percent))
                 abs_max = max_val if max_val is not None else 255 # Стандартный макс для яркости
                 abs_min = min_val if min_val is not None else 0   # Стандартный мин
                 final_value = int(abs_min + (abs_max - abs_min) * (percent / 100.0))
                 logger.info(f"Converted brightness {percent}% to absolute value {final_value} (range {abs_min}-{abs_max})")
             except ValueError:
                 return f"Sorry, couldn't parse percentage value '{value}' for brightness."

    elif cap_type == "binary":
        # parse_value уже вернул 'ON', 'OFF' или 'TOGGLE'
        allowed_bin = [capability_details.get("value_on", "ON"), capability_details.get("value_off", "OFF")]
        if capability_details.get("value_toggle"): allowed_bin.append(capability_details.get("value_toggle"))
        if parsed_value not in allowed_bin:
             return f"Sorry, invalid state '{parsed_value}' for '{capability_details['name']}'. Allowed: {allowed_bin}"

    elif cap_type == "enum":
        allowed_enum = capability_details.get("values")
        if not allowed_enum or parsed_value not in allowed_enum:
            return f"Sorry, invalid option '{parsed_value}' for '{capability_details['name']}'. Allowed options: {allowed_enum}"

    # TODO: Добавить обработку цвета (color) - это сложнее, т.к. может быть xy, hs, hex.
    # Потребуется конвертация из имени цвета ('red', 'blue') или температуры ('warm white')
    # в нужный формат (xy: [0.67, 0.32], color_temp: 400). Пока пропустим для простоты.
    if capability_details['name'] == 'color':
        logger.warning(f"Color setting for '{value}' is requested but not yet fully implemented in the tool.")
        # Здесь нужна логика конвертации value ('red', 'warm white', '#FF0000') в {"color": {"xy": [x, y]}} или {"color_temp": mired}
        # Пока просто вернем ошибку
        return f"Sorry, setting color by name ('{value}') is not fully supported yet."


    # 6. Формирование payload и отправка команды
    # Имя атрибута для MQTT payload - это оригинальное имя из exposes
    mqtt_attribute_name = capability_details['name']
    payload_dict = {mqtt_attribute_name: final_value}
    payload_json = json.dumps(payload_dict)

    topic = f"{settings.mqtt_broker.default_topic_base}/{target_device_name}/set"

    logger.info(f"Attempting to set attribute '{mqtt_attribute_name}' to '{final_value}' for device '{target_device_name}'. Topic: {topic}, Payload: {payload_json}")

    success = mqtt_client.schedule_publish(topic, payload_json, qos=1)

    if success:
        return f"Okay, set '{mqtt_attribute_name}' to '{final_value}' for device '{target_device_name}' (matched from '{user_description}')."
    else:
        return f"Sorry, failed to send the command for '{user_description}'."


# --- Убедитесь, что обновленный инструмент в списке ---
TOOLS: List = [
    set_device_attribute, # Используем обновленный инструмент
    say_to_console,
    manage_mem,
    search_mem
]