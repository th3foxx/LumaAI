from typing import List

# --- Импорт инструментов из модулей ВНУТРИ этого пакета 'tools' ---

# Инструменты из device_control.py
from .device_control import set_device_attribute

# Инструменты из memory.py (теперь внутри tools)
from .memory import manage_mem, search_mem

# Инструменты из scheduler.py (теперь внутри tools)
from .scheduler import schedule_reminder, list_reminders, cancel_reminder

# --- Агрегация всех инструментов в список TOOLS ---
TOOLS: List = [
    # Инструмент управления устройствами
    set_device_attribute,

    # Инструменты для работы с памятью
    manage_mem,
    search_mem,

    # Инструменты для работы с напоминаниями/расписанием
    schedule_reminder,
    list_reminders,
    cancel_reminder,
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
    "TOOLS",
]
