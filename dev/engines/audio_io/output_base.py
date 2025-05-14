from abc import ABC, abstractmethod
from typing import Dict, Any

class AudioOutputEngineBase(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    def play_tts_bytes(self, audio_bytes: bytes):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass
    
    @property
    @abstractmethod
    def is_running(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        pass

    async def startup(self):
        pass

    async def shutdown(self):
        pass