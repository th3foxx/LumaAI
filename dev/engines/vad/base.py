from abc import ABC, abstractmethod
from typing import List

class VADEngineBase(ABC):
    @abstractmethod
    def __init__(self, access_key: str, **kwargs):
        pass

    @abstractmethod
    def process(self, pcm: List[int]) -> float:
        """
        Processes a frame of audio to determine the probability of voice activity.
        Returns:
            float: Voice activity probability (0.0 to 1.0).
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

    async def startup(self):
        pass

    async def shutdown(self):
        pass