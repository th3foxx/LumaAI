import logging
from datetime import datetime
from typing import Optional

import pytz
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

DAYS_OF_WEEK_NOMINATIVE = [
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"
]
MONTHS_GENITIVE = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря"
]

def _get_friendly_timezone_name(tz_str: Optional[str], tz_object: Optional[pytz.BaseTzInfo]) -> str:
    """Возвращает дружелюбное имя часового пояса для голосового ответа."""
    if not tz_str and tz_object:
        local_tz_name = tz_object.tzname(datetime.now(tz_object))
        if local_tz_name and local_tz_name not in ['LMT', 'zzz'] and not local_tz_name.startswith(('+', '-')):
            if local_tz_name == "MSK":
                return "по Московскому времени"

            return f"по времени {local_tz_name}"
        return "по вашему местному времени" 

    if tz_str:
        if tz_str.lower() == 'utc':
            return "по Всемирному координированному времени (UTC)"
        if tz_str == 'Europe/Moscow':
            return "по Московскому времени"
        if tz_str == 'Europe/Kaliningrad':
            return "по Калининградскому времени"
        if tz_str == 'Europe/Samara':
            return "по Самарскому времени"
        if tz_str == 'Asia/Yekaterinburg':
            return "по Екатеринбургскому времени"

        parts = tz_str.split('/')
        if len(parts) > 1:
            city_or_region = parts[-1].replace('_', ' ')

            return f"по времени {city_or_region} (часовой пояс {tz_str})"
        return f"по времени часового пояса {tz_str}"
    return ""

def _format_time_for_speech(dt_object: datetime, friendly_tz_phrase: str) -> str:
    """Форматирует дату и время в разговорную строку."""
    hour = dt_object.hour
    minute = dt_object.minute

    if 0 <= hour <= 4:
        period = "ночи"
    elif 5 <= hour <= 11:
        period = "утра"
    elif 12 <= hour <= 16:
        period = "дня"
    else: # 17-23
        period = "вечера"

    time_str = f"{hour:02d}:{minute:02d}"

    if hour == 0 and minute == 0:
        time_spoken_intro = "Сейчас ровно полночь."
    elif hour == 12 and minute == 0:
        time_spoken_intro = "Сейчас ровно полдень."
    else:
        time_spoken_intro = f"Сейчас {time_str} {period}."

    day_of_week = DAYS_OF_WEEK_NOMINATIVE[dt_object.weekday()]
    day = dt_object.day
    month_name = MONTHS_GENITIVE[dt_object.month - 1]
    year = dt_object.year

    date_spoken = f"Сегодня {day_of_week}, {day}-е {month_name} {year} года."

    if friendly_tz_phrase:
        return f"{time_spoken_intro} {date_spoken}, {friendly_tz_phrase}."
    else:

        return f"{time_spoken_intro} {date_spoken}."


@tool
def get_current_time(timezone_str: Optional[str] = None) -> str:
    """
    Gets the current time, optionally for the specified time zone, formatted for a voice assistant.
    If no time zone is specified, returns the local time of the server.
    Time zones must be in tz database format (e.g. 'America/New_York', 'Europe/Moscow', 'Asia/Tokyo').
    A list of available timezones can be found, for example, with `pytz.all_timezones`.
    """
    try:
        actual_tz_object = None
        tz_str_for_friendly_name = timezone_str

        if timezone_str:
            try:
                tz = pytz.timezone(timezone_str)
                actual_tz_object = tz
            except pytz.UnknownTimeZoneError:
                logger.warning(f"Неизвестный часовой пояс: {timezone_str}")
                return (
                    f"К сожалению, я не знаю часовой пояс '{timezone_str}'. "
                    f"Попробуйте, например, 'Europe/Moscow' или 'America/New_York'."
                )
            now = datetime.now(tz)
        else:

            local_tz = datetime.now().astimezone().tzinfo
            actual_tz_object = local_tz

            now = datetime.now(local_tz)

        friendly_tz_phrase = _get_friendly_timezone_name(tz_str_for_friendly_name, actual_tz_object)
        return _format_time_for_speech(now, friendly_tz_phrase)

    except Exception as e:
        logger.error(f"Ошибка при получении времени: {e}", exc_info=True)
        return "Произошла ошибка, и я не смог узнать текущее время. Попробуйте еще раз чуть позже."