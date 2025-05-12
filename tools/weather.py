import os
import requests
import logging
from datetime import datetime
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
OPENWEATHERMAP_API_URL = "https://api.openweathermap.org/data/2.5/weather"

if not OPENWEATHERMAP_API_KEY:
    logger.warning(
        "OPENWEATHERMAP_API_KEY не установлен в переменных окружения. "
        "Инструмент погоды не будет работать."
    )


@tool
def get_current_weather(location: str, units: Optional[str] = "metric") -> str:
    """
    Gets the current weather for the specified location.

    Parameters:
        location (str): City and optionally country code (e.g. 'Moscow, RU', 'London, UK', 'Tokyo').
        units (str, optional): Units of measurement. 'metric' for Celsius, 'imperial' for Fahrenheit. Default 'metric'.
    """
    if not OPENWEATHERMAP_API_KEY:
        logger.error("API ключ для OpenWeatherMap не настроен.")
        return "Ошибка: Сервис погоды не настроен. API ключ отсутствует."
    if not location:
        return "Ошибка: Пожалуйста, укажите местоположение для получения прогноза погоды."
    if units not in ["metric", "imperial"]:
        logger.warning(f"Некорректные единицы измерения: {units}. Используется 'metric'.")
        units = "metric"

    params = {
        "q": location,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": units,
        "lang": "ru"
    }

    try:
        logger.info(f"Запрос погоды для {location} с параметрами: {params}")
        response = requests.get(OPENWEATHERMAP_API_URL, params=params, timeout=10) # Добавлен таймаут
        response.raise_for_status()  # Вызовет исключение для HTTP-ошибок (4xx или 5xx)
        data = response.json()
        logger.debug(f"Ответ от API погоды: {data}")

        main = data.get("main", {})
        weather_desc_list = data.get("weather", [{}])
        weather_desc = weather_desc_list[0].get("description", "нет данных") if weather_desc_list else "нет данных"
        
        temp = main.get("temp")
        feels_like = main.get("feels_like")
        humidity = main.get("humidity")
        wind_speed = data.get("wind", {}).get("speed")
        city_name = data.get("name", location)
        country = data.get("sys", {}).get("country", "")

        unit_symbol = "°C" if units == "metric" else "°F"
        speed_unit = "м/с" if units == "metric" else "миль/ч"

        if temp is None:
            logger.warning(f"Не удалось получить полные данные о погоде для {location}. Ответ API: {data}")
            return f"Не удалось получить полные данные о погоде для '{location}'. Возможно, местоположение указано неверно."

        return (
            f"Текущая погода в {city_name}{f', {country}' if country else ''}:\n"
            f"- Состояние: {weather_desc.capitalize()}\n"
            f"- Температура: {temp}{unit_symbol}\n"
            f"- Ощущается как: {feels_like}{unit_symbol}\n"
            f"- Влажность: {humidity}%\n"
            f"- Скорость ветра: {wind_speed} {speed_unit}"
        )

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logger.error("Ошибка авторизации с API OpenWeatherMap. Проверьте API ключ.")
            return "Ошибка: Недействительный API ключ для сервиса погоды."
        elif e.response.status_code == 404:
            logger.warning(f"Местоположение '{location}' не найдено API OpenWeatherMap.")
            return f"Ошибка: Не удалось найти погоду для местоположения '{location}'. Пожалуйста, проверьте написание или попробуйте другое местоположение."
        else:
            logger.error(f"HTTP ошибка при запросе погоды для {location}: {e.response.status_code} - {e.response.text}", exc_info=True)
            return f"Ошибка: Не удалось получить данные о погоде для {location} (код ошибки: {e.response.status_code})."
    except requests.exceptions.RequestException as e: # Общий класс для ошибок requests (включая таймауты, проблемы с DNS и т.д.)
        logger.error(f"Сетевая ошибка при запросе погоды для {location}: {e}", exc_info=True)
        return f"Ошибка: Проблема с сетью при попытке получить погоду для {location}."
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении погоды для {location}: {e}", exc_info=True)
        return f"Извините, произошла неожиданная ошибка при получении погоды для {location}."