# --- START OF FILE tools.py ---

from typing import List, Literal, Optional # Добавили Optional
import json
import logging
from thefuzz import process as fuzz_process, fuzz # Добавили fuzz для прямого сравнения

from langchain_core.tools import tool
from mqtt_client import mqtt_client # Import the global instance
from settings import settings # Import settings for topic base

logger = logging.getLogger(__name__)

# --- Предыдущие инструменты остаются без изменений ---
@tool
def manage_mem(operation: str, data: str) -> str:
    """Placeholder for memory management."""
    return f"[Memory '{operation}' with data: {data}]"

@tool
def search_mem(query: str) -> str:
    """Placeholder for memory search."""
    return f"[Memory search results for '{query}']"

@tool
def search_web(query: str) -> str:
    """Search the web for fresh data (placeholder)."""
    return f"[Search results for '{query}']"

@tool
def say_to_console(text: str) -> str:
    """Outputs a text to the console"""
    logger.info(f"[Tool Output] Console: {text}")
    return f"Okay, printed '{text}' to console."


# --- Обновленный инструмент control_device ---
@tool
def control_device(
    user_description: str,
    state: Literal["ON", "OFF"],
    guessed_friendly_name: Optional[str] = None
) -> str:
    """
    Controls a Zigbee device (like lights or switches) based on a natural language description.
    Use 'ON' to turn the device on, 'OFF' to turn it off.

    Parameters:
        user_description (str): The user's description of the device (e.g., 'light in the hall', 'kitchen lamp').
        state (Literal["ON", "OFF"]): The desired state.
        guessed_friendly_name (Optional[str]): Your (the LLM's) best guess for the exact 'friendly_name' of the device based on the user_description and context. Provide this if you are reasonably confident. Leave empty or null if unsure.
                                               Example: If user says 'turn off the living room ceiling light', you might guess 'living_room_ceiling_1'.

    Example call: control_device(user_description='свет в зале', state='OFF', guessed_friendly_name='лампа-зал')
    Example call (unsure): control_device(user_description='that lamp over there', state='ON')
    """
    if not user_description:
        return "Error: You must provide a description of the device."

    # 1. Получаем актуальный список устройств
    available_devices = mqtt_client.get_device_friendly_names()
    if not available_devices:
        logger.error("Cannot control device: No device list available from MQTT client.")
        return "Sorry, I can't control devices right now. The list of devices is unavailable. Please try again later."

    target_device_name = None
    match_method = "None" # Для логирования, как было найдено совпадение

    # 2. Проверяем предположение LLM, если оно есть
    if guessed_friendly_name:
        logger.info(f"LLM guessed friendly_name: '{guessed_friendly_name}'. Validating against: {available_devices}")
        # Ищем совпадение для предположения LLM. Используем высокий порог для уверенности.
        # process.extractOne может быть недостаточно строгим, попробуем найти почти точное совпадение
        exactish_match = None
        highest_score = 0
        # Порог для "почти точного" совпадения с догадкой LLM
        guess_validation_threshold = 90

        for device in available_devices:
            # Используем fuzz.ratio или fuzz.token_sort_ratio для сравнения
            score = fuzz.token_sort_ratio(guessed_friendly_name, device)
            if score >= guess_validation_threshold and score > highest_score:
                highest_score = score
                exactish_match = device

        if exactish_match:
            target_device_name = exactish_match
            match_method = f"LLM guess validated (score: {highest_score})"
            logger.info(f"Validated LLM guess '{guessed_friendly_name}' as '{target_device_name}' with score {highest_score}")
        else:
            logger.warning(f"LLM guess '{guessed_friendly_name}' not found with high confidence (>={guess_validation_threshold}) in available devices. Falling back to user description matching.")
            # Если догадка не подтвердилась, сбрасываем ее и ищем по описанию

    # 3. Если догадка не сработала или не была предоставлена, ищем по описанию пользователя
    if not target_device_name:
        logger.info(f"Matching user description '{user_description}' against: {available_devices}")
        # Используем extractOne с разумным порогом для описания
        description_match_threshold = 75 # Можно настроить
        best_match_tuple = fuzz_process.extractOne(user_description, available_devices, score_cutoff=description_match_threshold)

        if best_match_tuple:
            matched_name, match_score = best_match_tuple
            target_device_name = matched_name
            match_method = f"User description matched (score: {match_score})"
            logger.info(f"Matched user description '{user_description}' to device '{target_device_name}' with score {match_score}")
        else:
            logger.warning(f"Could not find a suitable match for description '{user_description}' (threshold: {description_match_threshold}) among devices: {available_devices}")
            return f"Sorry, I couldn't find a device matching '{user_description}' well enough. Can you be more specific or check the device name?"

    # 4. Если устройство найдено (любым способом)
    if target_device_name:
        topic = f"{settings.mqtt_broker.default_topic_base}/{target_device_name}/set"
        payload_dict = {"state": state}
        payload_json = json.dumps(payload_dict)

        logger.info(f"Attempting to control target device '{target_device_name}' (found via: {match_method}). Topic: {topic}, Payload: {payload_json}")

        success = mqtt_client.schedule_publish(topic, payload_json, qos=1)

        if success:
            # Возвращаем подтверждение, возможно, указывая найденное устройство для ясности
            return f"Okay, command sent to turn {state.lower()} '{target_device_name}' (matched from '{user_description}')."
        else:
            logger.error(f"Failed to schedule MQTT command for target device '{target_device_name}'.")
            return f"Sorry, I couldn't send the command for '{user_description}'. There might be a connection issue with the device control system."
    else:
        # Эта ветка не должна достигаться из-за проверок выше, но на всякий случай
        logger.error(f"Internal logic error: No target device name found for '{user_description}' after checks.")
        return f"Sorry, an internal error occurred while trying to find the device for '{user_description}'."


# --- Убедитесь, что обновленный инструмент в списке ---
TOOLS: List = [
    search_web,
    control_device, # Используем обновленный инструмент
    say_to_console,
    manage_mem,
    search_mem
]

# --- END OF FILE tools.py ---