from typing import List

# --- Импорт инструментов из модулей ВНУТРИ этого пакета 'tools' ---

from .memory import add_personal_memory, search_personal_memories, initialize_memory_tools
from .device_control import set_device_attribute
from .scheduler import schedule_reminder, list_reminders, cancel_reminder
from .weather import get_current_weather
from .web_search import search_web_serper
from .time import get_current_time
from .communication import send_email_to_contact, send_telegram_message_to_contact, list_contacts
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
    add_personal_memory,
    search_personal_memories,
    set_device_attribute,
    schedule_reminder,
    list_reminders,
    cancel_reminder,
    get_current_time,
    get_current_weather,
    send_email_to_contact,
    send_telegram_message_to_contact,
    list_contacts,
    search_web_serper,
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
    "add_personal_memory",
    "search_personal_memories",
    "initialize_memory_tools",
    "set_device_attribute",
    "schedule_reminder",
    "list_reminders",
    "cancel_reminder",
    "get_current_time",
    "get_current_weather",
    "send_email_to_contact",
    "send_telegram_message_to_contact",
    "list_contacts",
    "search_web_serper",
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
