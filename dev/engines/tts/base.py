from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any

class TTSEngineBase(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Synthesizes speech from text and yields audio chunks.
        Must yield an empty bytes object or raise StopAsyncIteration when done.
        """
        # Example:
        # yield b"audio_chunk_1"
        # yield b"audio_chunk_2"
        # yield b"" # Indicates end of stream if necessary, or just StopAsyncIteration
        if False: # This is to make it an async generator
            yield b""


    @abstractmethod
    async def synthesize_once(self, text: str) -> bytes:
        """Synthesizes speech and returns all audio bytes at once."""
        pass

    @abstractmethod
    async def startup(self):
        """Starts any necessary server or initializes resources."""
        pass

    @abstractmethod
    async def shutdown(self):
        """Stops any server and releases resources."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Checks if the TTS engine/server is operational."""
        pass