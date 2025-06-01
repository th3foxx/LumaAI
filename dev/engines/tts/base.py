from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional

class TTSEngineBase(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Synthesizes speech from text and yields audio chunks.
        The generator should simply stop (raising StopAsyncIteration implicitly)
        when all audio chunks have been yielded.
        """
        # Example:
        # yield b"audio_chunk_1"
        # yield b"audio_chunk_2"
        # ... (no more chunks to yield, function ends, StopAsyncIteration is raised)
        if False: # This is to make it an async generator for type checking if no concrete implementation
            yield b"" # pragma: no cover (or similar if you use coverage tools)

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

    @abstractmethod
    def get_output_sample_rate(self) -> int:
        """
        Returns the native output sample rate of this TTS engine in Hz.
        For hybrid engines, this might be a preferred or default rate.
        It's recommended to get the active sub-engine (e.g., via
        HybridTTSEngine.get_active_engine_for_synthesis()) and query its
        sample rate directly for more accuracy when using a hybrid engine.
        """
        pass