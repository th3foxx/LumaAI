from abc import ABC, abstractmethod
import json

class STTEngineBase(ABC):
    class RecognizerInstance(ABC):
        @abstractmethod
        def accept_waveform(self, audio_chunk: bytes) -> bool:
            pass

        @abstractmethod
        def partial_result(self) -> str: # Returns JSON string
            pass

        @abstractmethod
        def final_result(self) -> str: # Returns JSON string
            pass
        
        @abstractmethod
        def result(self) -> str: # Convenience for non-streaming if needed
            pass

        @abstractmethod
        def reset(self):
            pass

    @abstractmethod
    def __init__(self, model_path: str, sample_rate: int, **kwargs):
        pass

    @abstractmethod
    def create_recognizer(self) -> 'STTEngineBase.RecognizerInstance':
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        pass

    @abstractmethod
    def delete(self): # To release the main model if necessary
        pass

    async def startup(self):
        pass

    async def shutdown(self):
        pass