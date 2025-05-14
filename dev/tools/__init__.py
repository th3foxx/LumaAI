from typing import List

# --- Импорт инструментов из модулей ВНУТРИ этого пакета 'tools' ---

from .device_control import set_device_attribute
from .memory import manage_mem, search_mem
from .scheduler import schedule_reminder, list_reminders, cancel_reminder
from .weather import get_current_weather
from .time import get_current_time
from .music_control import (
    play_music,
    play_from_youtube,
    pause_music,
    resume_music,
    stop_music,
    next_song,
    previous_song,
    set_volume,
    get_current_song,
    like_current_song,
    unlike_current_song,
    play_liked_songs,
    list_liked_songs
)

# --- Агрегация всех инструментов в список TOOLS ---
TOOLS: List = [
    set_device_attribute,
    manage_mem,
    search_mem,
    schedule_reminder,
    list_reminders,
    cancel_reminder,
    get_current_time,
    get_current_weather,
    play_music,
    play_from_youtube,
    pause_music,
    resume_music,
    stop_music,
    next_song,
    previous_song,
    set_volume,
    get_current_song,
    like_current_song,
    unlike_current_song,
    play_liked_songs,
    list_liked_songs
]

# --- Опционально: Определение __all__ для контроля импорта через * ---
# Это позволяет делать from tools import * (хотя это обычно не рекомендуется)
# и также помогает IDE с автодополнением при импорте from tools import ...
__all__ = [
    "set_device_attribute",
    "manage_mem",
    "search_mem",
    "schedule_reminder",
    "list_reminders",
    "cancel_reminder",
    "get_current_time",
    "get_current_weather",
    "play_music",
    "play_from_youtube",
    "pause_music",
    "resume_music",
    "stop_music",
    "next_song",
    "previous_song",
    "set_volume",
    "get_current_song",
    "like_current_song",
    "unlike_current_song",
    "play_liked_songs",
    "list_liked_songs",
    "TOOLS",
]
