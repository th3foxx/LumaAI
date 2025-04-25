from typing import List, Literal

from langchain_core.tools import tool
from .memory import manage_mem, search_mem


@tool
def search_web(query: str) -> str:
    """Search the web for fresh data (placeholder)."""
    return f"[Search results for '{query}']"


@tool
def say_to_console(text: str) -> str:
    """Outputs a text to the console"""
    return f"[Console: {text}]"


@tool
def light_control(action: Literal["on", "off"], location: str) -> str:
    """Control smart lights: 'on' or 'off' in a given location (placeholder)."""
    print(f"Turning lights {action} in {location}...")
    return f"Lights turned {action} in {location}."


TOOLS: List = [search_web, light_control,say_to_console, manage_mem, search_mem]