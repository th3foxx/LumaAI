from abc import ABC, abstractmethod
from typing import Callable, Coroutine, Any

ProcessAudioCallback = Callable[[bytes], Coroutine[Any, Any, None]]

class AudioInputEngineBase(ABC):
    @abstractmethod
    def __init__(self, process_audio_callback: ProcessAudioCallback, loop: Any, config: Dict[str, Any]):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def pause(self):
        pass

    @abstractmethod
    def resume(self):
        pass
    
    @property
    @abstractmethod
    def is_running(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        pass
    
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass

    @property
    @abstractmethod
    def frame_length(self) -> int: # samples per frame
        pass
    
    @property
    @abstractmethod
    def channels(self) -> int:
        pass

    async def startup(self): # For consistency, though typically sync for audio devices
        pass

    async def shutdown(self):
        pass