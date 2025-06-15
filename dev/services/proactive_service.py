import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple

from app import globals as G
from settings import settings
from connectivity import is_internet_available # <-- НОВЫЙ ИМПОРТ

logger = logging.getLogger(__name__)

# --- Конфигурация и состояние (без изменений) ---
PROACTIVE_CHECK_INTERVAL = 60 * 5
MIN_COOLDOWN_BETWEEN_EVENTS = timedelta(minutes=20)
_proactive_task: Optional[asyncio.Task] = None
_stop_event = asyncio.Event()
_last_proactive_event_time: Optional[datetime] = None
_morning_briefing_done_for_day: Optional[int] = None

# --- Вспомогательные функции ---
async def _get_device_state(device_friendly_name: str) -> Optional[dict]:
    if G.comm_service and hasattr(G.comm_service, 'get_device_state'):
        return await G.comm_service.get_device_state(device_friendly_name)
    return None

# --- НОВАЯ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ---
async def _get_user_location_from_memory() -> str:
    """
    Пытается получить местоположение пользователя из Mem0.
    Возвращает найденный город или город по умолчанию.
    """
    default_location = settings.tools.default_weather_location
    
    if not G.mem0_client:
        return default_location # Память не инициализирована

    try:
        # Импортируем инструмент здесь, чтобы избежать циклических зависимостей
        from tools.memory import search_personal_memories
        
        # Ищем факты, связанные с местоположением
        search_results = await search_personal_memories.ainvoke({"query": "местоположение пользователя, город, где живет пользователь"})
        
        if "No relevant information found" in search_results or "nothing with high confidence" in search_results:
            logger.debug("No location found in Mem0, using default.")
            return default_location

        # Простая логика извлечения города. Можно улучшить с помощью LLM или регэкспов.
        # Ищем строки вида "живет в X", "город Y", "находится в Z"
        lines = search_results.split('\n')
        for line in lines:
            line_lower = line.lower()
            if "живет в" in line_lower:
                # "Живет в Санкт-Петербурге" -> "Санкт-Петербурге"
                city = line.split(" в ")[-1].strip()
                logger.info(f"User location found in Mem0: '{city}'")
                return city
            if "город" in line_lower:
                city = line.split(" ", 1)[-1].strip()
                logger.info(f"User location found in Mem0: '{city}'")
                return city

    except Exception as e:
        logger.error(f"Failed to get user location from memory: {e}", exc_info=True)
    
    return default_location


# --- Обновленные функции-проверки ---

async def _check_morning_briefing() -> Optional[str]:
    """Генерирует триггер для утреннего приветствия."""
    global _morning_briefing_done_for_day
    now = datetime.now()
    if not (7 <= now.hour < 10): return None
    day_of_year = now.timetuple().tm_yday
    if _morning_briefing_done_for_day == day_of_year: return None
    
    from tools.weather import get_current_weather
    from tools.scheduler import get_pending_reminders_from_db

    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
    user_location = await _get_user_location_from_memory()
    weather_info = await get_current_weather.ainvoke({"location": user_location})
    
    reminders = await get_pending_reminders_from_db()
    today_reminders_count = sum(1 for r in reminders if datetime.fromisoformat(r['due_time_utc'].replace('Z', '+00:00')).astimezone().date() == now.date())

    weather_context = "unknown"
    if "Ошибка" not in weather_info:
        # Извлекаем название города из ответа API, чтобы оно было в правильном падеже
        city_name_from_api = weather_info.splitlines()[0].split(' ')[3].replace(':', '')
        weather_context = f"in {city_name_from_api} is {weather_info.splitlines()[1].split(': ')[1].lower()}"
    else:
        weather_context = f"in {user_location} is unknown"


    _morning_briefing_done_for_day = day_of_year
    return f"[PROACTIVE_TRIGGER] Event: MORNING_BRIEFING. Context: Weather {weather_context}, {today_reminders_count} pending reminders for today."


async def _check_evening_lights() -> Optional[str]:
    """Генерирует триггер для предложения включить свет."""
    now = datetime.now()
    if not (19 <= now.hour < 22): return None
    main_light_name = settings.proactive.main_evening_light
    light_state_data = await _get_device_state(main_light_name)
    if not light_state_data or light_state_data.get("state") == "ON": return None
    
    return f"[PROACTIVE_TRIGGER] Event: EVENING_LIGHTS_OFF. Context: It's {now.strftime('%H:%M')}, '{main_light_name}' is off."


async def _check_weather_based_suggestion() -> Optional[str]:
    """
    Проверяет погоду и безусловно передает ее LLM для принятия решения.
    Python-код больше не решает, "скучная" погода или нет.
    """
    from tools.weather import get_current_weather
    now = datetime.now()

    # Оставляем только временное окно, чтобы не беспокоить пользователя ночью.
    if not (12 <= now.hour < 22):
        return None

    # Получаем местоположение и погоду
    user_location = await _get_user_location_from_memory()
    weather_info_raw = await get_current_weather.ainvoke({"location": user_location})

    if "Ошибка" in weather_info_raw:
        return None # Если не удалось получить погоду, ничего не делаем

    # Формируем триггер с сырыми данными. Никаких фильтров.
    # LLM получит это и сам решит, что делать.
    return (
        f"[PROACTIVE_TRIGGER] Event: WEATHER_CONTEXT_FOR_ANALYSIS. "
        f"Context: Current weather in {user_location} is as follows:\n{weather_info_raw}"
    )

async def _check_idle_chatter() -> Optional[str]:
    """Генерирует триггер для начала случайного диалога."""
    return "[PROACTIVE_TRIGGER] Event: IDLE_CHATTER. Context: User has been silent for a while."


# --- ОСНОВНОЙ ЦИКЛ (ОБНОВЛЕН) ---

async def proactive_checking_loop():
    global _last_proactive_event_time
    logger.info("Proactive Service started.")
    await asyncio.sleep(20)

    while not _stop_event.is_set():
        try:
            # 1. --- НОВОЕ: Главная проверка на онлайн-режим ---
            if not await is_internet_available():
                logger.debug("Proactive service sleeping: no internet connection.")
                await asyncio.sleep(PROACTIVE_CHECK_INTERVAL)
                continue

            # 2. Проверка кулдауна
            if _last_proactive_event_time and (datetime.now() - _last_proactive_event_time < MIN_COOLDOWN_BETWEEN_EVENTS):
                await asyncio.sleep(PROACTIVE_CHECK_INTERVAL)
                continue

            # 3. Проверка, не занят ли ассистент
            if G.manager and G.manager.state != "wakeword":
                logger.debug(f"Proactive service sleeping: manager is busy (state: {G.manager.state}).")
                await asyncio.sleep(60) # Если занят, проверяем чаще
                continue

            proactive_checks = [
                _check_morning_briefing,
                _check_evening_lights,
                _check_weather_based_suggestion,
                _check_idle_chatter,
            ]
            random.shuffle(proactive_checks)

            for check_func in proactive_checks:
                trigger_message = await check_func()
                if trigger_message:
                    logger.info(f"Proactive trigger fired by {check_func.__name__}. Trigger: '{trigger_message}'")
                    
                    if G.manager:
                        # --- ИЗМЕНЕНО: Вызываем новый метод в ConnectionManager ---
                        await G.manager.initiate_proactive_dialogue(trigger_message)
                        _last_proactive_event_time = datetime.now()
                        break 
        except Exception as e:
            logger.error(f"Critical error in proactive_checking_loop: {e}", exc_info=True)
        
        await asyncio.sleep(PROACTIVE_CHECK_INTERVAL)

async def start_proactive_service():
    global _proactive_task, _stop_event
    if _proactive_task is None or _proactive_task.done():
        _stop_event.clear()
        _proactive_task = asyncio.create_task(proactive_checking_loop())
        logger.info("Proactive service background task started.")

async def stop_proactive_service():
    global _proactive_task
    if _proactive_task and not _proactive_task.done():
        logger.info("Stopping proactive service background task...")
        _stop_event.set()
        _proactive_task.cancel()
        try:
            await _proactive_task
        except asyncio.CancelledError:
            pass
        logger.info("Proactive service background task stopped.")
    _proactive_task = None