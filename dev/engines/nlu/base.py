from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class NLUEngineBase(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    async def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parses text to extract intent and entities for offline commands.
        """
        pass

    async def startup(self):
        pass

    async def shutdown(self):
        pass