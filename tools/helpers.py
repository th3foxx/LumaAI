import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

def normalize_text_key(name: str) -> str:
    """Normalizes text key to lowercase and strips whitespace."""
    return name.lower().strip()

def normalize_attribute_name(attr_name: str) -> str:
    """Приводит имя атрибута к нижнему регистру и заменяет пробелы на подчеркивания."""
    return attr_name.lower().replace(" ", "_")

def parse_value(value_str: str, target_type: str) -> Optional[Any]:
    """
    Пытается преобразовать строку значения в нужный тип (int, float, bool, str).
    Excludes preset logic which is handled specifically in the tool using this.
    """
    try:
        if target_type == "numeric":
             # Пытаемся сначала в int, потом в float
             try: return int(value_str)
             except ValueError: return float(value_str)
        elif target_type == "binary":
            # Приводим к стандартным ON/OFF для простоты
            val_lower = value_str.strip().lower()
            # Расширяем синонимы
            if val_lower in ["on", "вкл", "включить", "включи", "1", "true", "да"]: return "ON"
            if val_lower in ["off", "выкл", "выключить", "выключи", "0", "false", "нет"]: return "OFF"
            if val_lower in ["toggle", "переключить", "переключи"]: return "TOGGLE"
            return None # Не распознано
        elif target_type == "enum":
            return value_str # Возвращаем как есть, проверка будет по списку
        else: # По умолчанию считаем строкой
            return value_str
    except (ValueError, TypeError):
        logger.warning(f"Could not parse value '{value_str}' as {target_type}")
        return None