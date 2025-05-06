# offline_controller.py
import logging
import json
from typing import Dict, Optional, List, Any

from thefuzz import process as fuzz_process, fuzz
import httpx

# --- ИМПОРТИРУЕМ ОСНОВНОЙ ИНСТРУМЕНТ ---
from tools.device_control import set_device_attribute
# ----------------------------------------
from mqtt_client import mqtt_client
from settings import settings

logger = logging.getLogger(__name__)

# --- State ---
_last_mentioned_device: Optional[str] = None

# --- Rasa NLU API Configuration ---
RASA_NLU_URL = settings.rasa_nlu.url
RASA_NLU_TIMEOUT = settings.rasa_nlu.timeout
RASA_INTENT_CONFIDENCE_THRESHOLD = settings.rasa_nlu.intent_confidence_threshold

# --- Helper Functions ---
def find_best_match(query: str, choices: List[str], scorer=fuzz.token_sort_ratio, score_cutoff=75) -> Optional[str]:
    # ... (остается без изменений) ...
    if not choices:
        return None
    match = fuzz_process.extractOne(query, choices, scorer=scorer, score_cutoff=score_cutoff)
    return match[0] if match else None

# --- УДАЛЕНА ФУНКЦИЯ parse_value_offline ---

async def parse_offline_command(text: str) -> Optional[Dict[str, Any]]:
    """
    Parses user text via Rasa NLU and prepares arguments for set_device_attribute tool.
    """
    global _last_mentioned_device
    payload = {"text": text}
    logger.debug(f"Attempting offline parse via Rasa NLU API (URL: {RASA_NLU_URL}) for: '{text}'")

    if not RASA_NLU_URL:
        logger.error("Rasa NLU URL is not configured. Cannot parse offline command.")
        return None

    try:
        async with httpx.AsyncClient(timeout=RASA_NLU_TIMEOUT) as client:
            response = await client.post(RASA_NLU_URL, json=payload)
            response.raise_for_status()
            result = response.json()
        logger.debug(f"Rasa NLU API raw result: {json.dumps(result, indent=2, ensure_ascii=False)}")

        intent_data = result.get("intent", {})
        intent = intent_data.get("name")
        confidence = intent_data.get("confidence", 0.0)
        entities = result.get("entities", [])

        if not intent or confidence < RASA_INTENT_CONFIDENCE_THRESHOLD:
            logger.warning(f"NLU confidence low ({confidence:.2f} < {RASA_INTENT_CONFIDENCE_THRESHOLD}). Text: '{text}'.")
            return None # Недостаточно уверенности или нет интента

        logger.info(f"Rasa NLU classified intent as '{intent}' with confidence {confidence:.2f}")

        # --- Извлечение сущностей (как и раньше) ---
        device_parts = [e['value'] for e in entities if e['entity'] in ['device_description', 'location']]
        device_search_text = " ".join(device_parts).strip()
        pronoun_entity = next((e for e in entities if e['entity'] == 'pronoun_device'), None)
        attribute_name_entity = next((e for e in entities if e['entity'] == 'attribute_name'), None)
        attribute_value_entities = [e['value'] for e in entities if e['entity'] == 'attribute_value']
        attribute_value_raw = " ".join(attribute_value_entities).strip() if attribute_value_entities else None

        logger.debug(f"Extracted Entities: device_search='{device_search_text}', pronoun='{pronoun_entity['value'] if pronoun_entity else None}', attr_name='{attribute_name_entity['value'] if attribute_name_entity else None}', attr_value='{attribute_value_raw}'")

        # --- Разрешение устройства (как и раньше) ---
        available_devices = mqtt_client.get_device_friendly_names()
        if not available_devices:
             logger.error("MQTT device list unavailable for offline command.")
             return None

        resolved_device_name = None
        # ... (логика поиска по device_search_text, pronoun_entity, _last_mentioned_device) ...
        # Пример:
        if device_search_text:
            matched = find_best_match(device_search_text, available_devices, score_cutoff=70)
            if matched:
                resolved_device_name = matched
                _last_mentioned_device = matched
        elif pronoun_entity and _last_mentioned_device and _last_mentioned_device in available_devices:
             resolved_device_name = _last_mentioned_device
        elif intent in ['set_attribute', 'activate_device', 'deactivate_device'] and _last_mentioned_device and _last_mentioned_device in available_devices:
             # Используем контекст, если устройство не указано явно
             resolved_device_name = _last_mentioned_device

        if not resolved_device_name:
            logger.error(f"Failed to resolve target device for text: '{text}'.")
            return None

        # --- Определение атрибута и значения для set_device_attribute ---
        target_attribute = None
        target_value_string = None

        if intent == "activate_device":
            target_attribute = "state"
            target_value_string = "ON"
        elif intent == "deactivate_device":
            target_attribute = "state"
            target_value_string = "OFF"
        elif intent == "set_attribute":
            # Определяем атрибут (как и раньше, через сущность или угадывание)
            if attribute_name_entity:
                # Нормализация имени атрибута из NLU
                norm_attr_from_nlu = attribute_name_entity['value'].lower().strip() # Простая нормализация
                if norm_attr_from_nlu in ["яркость", "brightness"]: target_attribute = "brightness"
                elif norm_attr_from_nlu in ["цвет", "color"]: target_attribute = "color"
                elif norm_attr_from_nlu in ["температура", "temperature", "цветовая температура", "цвет температура"]: target_attribute = "color_temp"
                # ... другие атрибуты ...
                else: logger.warning(f"NLU extracted unknown attribute name: '{attribute_name_entity['value']}'")
            elif attribute_value_raw:
                # Попытка угадать атрибут по значению (если нужно)
                # Эта логика может быть сложной и менее надежной, возможно, лучше требовать сущность атрибута
                 logger.debug(f"Attribute name missing, value is '{attribute_value_raw}'. Guessing is complex, relying on NLU attribute entity is preferred.")
                 # Можно оставить угадывание или убрать его

            if not target_attribute:
                 logger.error(f"Failed to determine target attribute for 'set_attribute'. Text: '{text}'")
                 return None

            # Значение - это просто сырая строка из NLU
            if attribute_value_raw is None:
                 logger.error(f"Intent is 'set_attribute', but no attribute_value entity extracted. Text: '{text}'")
                 return None
            target_value_string = attribute_value_raw

        else:
            logger.warning(f"Unhandled intent '{intent}' from NLU. Text: '{text}'")
            return None # Не команда управления

        # --- Проверка наличия всего необходимого ---
        if not resolved_device_name or not target_attribute or target_value_string is None:
            logger.error(f"Command parsing incomplete. Device: {resolved_device_name}, Attr: {target_attribute}, Value: {target_value_string}. Text: '{text}'")
            return None

        # --- Формирование результата для вызова set_device_attribute ---
        command_args = {
            "user_description": text, # Передаем оригинальный текст
            "attribute": target_attribute,
            "value": target_value_string, # Передаем строку как есть
            "guessed_friendly_name": resolved_device_name # Передаем найденное имя
        }
        logger.info(f"Offline command parsed for tool execution: {command_args}")
        return command_args

    # ... (обработка исключений httpx, json и т.д. остается) ...
    except Exception as e:
        logger.error(f"Unexpected error during offline command parsing: {e}", exc_info=True)
        return None


async def execute_offline_command(command_args: Optional[Dict[str, Any]]) -> str:
    """
    Executes the command by calling the set_device_attribute tool's invoke method.
    """
    if command_args is None:
        logger.warning("execute_offline_command called with None arguments.")
        return "Sorry, I couldn't understand that command."

    # Аргументы уже подготовлены в parse_offline_command
    logger.info(f"Executing offline command via set_device_attribute tool with args: {command_args}")

    try:
        # --- ИСПОЛЬЗУЕМ МЕТОД .invoke() ИНСТРУМЕНТА ---
        # Передаем словарь аргументов как единственный параметр в invoke()
        result_message = set_device_attribute.invoke(command_args)
        # ---------------------------------------------
        logger.info(f"set_device_attribute tool returned: {result_message}")
        return result_message
    except Exception as e:
        # Ловим ошибки, которые мог выбросить сам инструмент set_device_attribute
        logger.error(f"Error executing command via set_device_attribute tool: {e}", exc_info=True)
        # Формируем сообщение об ошибке, используя данные из command_args
        device_name = command_args.get('guessed_friendly_name', 'the device')
        attr_name = command_args.get('attribute', 'attribute')
        # Можно сделать сообщение более конкретным, если тип ошибки известен
        return f"Sorry, an error occurred while trying to set {attr_name} for {device_name}."