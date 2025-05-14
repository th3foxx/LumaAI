from typing import List

# --- Импорт инструментов из модулей ВНУТРИ этого пакета 'tools' ---

from .device_control import set_device_attribute
from .memory import manage_mem, search_mem
from .scheduler import schedule_reminder, list_reminders, cancel_reminder
from .time import get_current_time

# --- Агрегация всех инструментов в список TOOLS ---
TOOLS: List = [
    set_device_attribute,
    manage_mem,
    search_mem,
    schedule_reminder,
    list_reminders,
    cancel_reminder,
    get_current_time
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
    "get_current_time"
    "TOOLS",
]
