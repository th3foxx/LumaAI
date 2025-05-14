import pvporcupine
from typing import List
import logging
from .base import WakeWordEngineBase

logger = logging.getLogger(__name__)

class PicovoicePorcupineEngine(WakeWordEngineBase):
    def __init__(self, access_key: str, keywords: List[str], sensitivities: List[float] = None, 
                 model_path: str = None, library_path: str = None):
        try:
            self._porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=keywords,
                sensitivities=sensitivities,
                model_path=model_path,
                library_path=library_path
            )
            logger.info(f"Porcupine initialized. Frame: {self.frame_length}, Rate: {self.sample_rate}, Version: {self.version}")
        except pvporcupine.PorcupineError as e:
            logger.error(f"Porcupine initialization failed: {e}", exc_info=True)
            raise

    def process(self, pcm: List[int]) -> int:
        try:
            return self._porcupine.process(pcm)
        except pvporcupine.PorcupineError as e:
            logger.error(f"Porcupine processing error: {e}")
            # Depending on desired behavior, could raise or return -1
            return -1 
            
    @property
    def frame_length(self) -> int:
        return self._porcupine.frame_length

    @property
    def sample_rate(self) -> int:
        return self._porcupine.sample_rate

    @property
    def version(self) -> str:
        return self._porcupine.version

    def delete(self):
        if hasattr(self, '_porcupine') and self._porcupine:
            self._porcupine.delete()
            logger.info("Porcupine instance deleted.")

    async def shutdown(self):
        self.delete()