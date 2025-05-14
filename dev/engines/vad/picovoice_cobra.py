import pvcobra
from typing import List
import logging
from .base import VADEngineBase

logger = logging.getLogger(__name__)

class PicovoiceCobraEngine(VADEngineBase):
    def __init__(self, access_key: str, library_path: str = None):
        try:
            self._cobra = pvcobra.create(
                access_key=access_key,
                library_path=library_path
            )
            logger.info(f"Cobra VAD initialized. Frame: {self.frame_length}, Rate: {self.sample_rate}, Version: {self.version}")
        except pvcobra.CobraError as e:
            logger.error(f"Cobra VAD initialization failed: {e}", exc_info=True)
            raise

    def process(self, pcm: List[int]) -> float:
        try:
            return self._cobra.process(pcm)
        except pvcobra.CobraError as e:
            logger.error(f"Cobra processing error: {e}")
            return 0.0 # Return low probability on error

    @property
    def frame_length(self) -> int:
        return self._cobra.frame_length

    @property
    def sample_rate(self) -> int:
        return self._cobra.sample_rate

    @property
    def version(self) -> str:
        return self._cobra.version

    def delete(self):
        if hasattr(self, '_cobra') and self._cobra:
            self._cobra.delete()
            logger.info("Cobra instance deleted.")
    
    async def shutdown(self):
        self.delete()