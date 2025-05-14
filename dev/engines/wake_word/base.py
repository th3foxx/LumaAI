from abc import ABC, abstractmethod
from typing import List

class WakeWordEngineBase(ABC):
    @abstractmethod
    def __init__(self, access_key: str, keywords: List[str], sensitivities: List[float] = None, **kwargs):
        pass

    @abstractmethod
    def process(self, pcm: List[int]) -> int:
        """
        Processes a frame of audio looking for the wake word.
        Returns:
            int: Index of the detected keyword, or -1 if no keyword is detected.
        """
        pass

    @property
    @abstractmethod
    def frame_length(self) -> int:
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @abstractmethod
    def delete(self):
        pass

    async def startup(self): # Optional, for engines needing async setup
        pass

    async def shutdown(self): # Optional, for engines needing async teardown
        pass