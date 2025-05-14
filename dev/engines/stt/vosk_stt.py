import os
import json
import logging
from vosk import Model as VoskModel, KaldiRecognizer
from .base import STTEngineBase

logger = logging.getLogger(__name__)

class VoskSTTEngine(STTEngineBase):
    class VoskRecognizerInstance(STTEngineBase.RecognizerInstance):
        def __init__(self, vosk_model: VoskModel, sample_rate: int):
            self._recognizer = KaldiRecognizer(vosk_model, sample_rate)
            self._recognizer.SetWords(False) # Or True if word timings are needed

        def accept_waveform(self, audio_chunk: bytes) -> bool:
            return self._recognizer.AcceptWaveform(audio_chunk)

        def partial_result(self) -> str:
            return self._recognizer.PartialResult()

        def final_result(self) -> str:
            return self._recognizer.FinalResult()
        
        def result(self) -> str: # For non-streaming if vosk_model supports it directly
            return self._recognizer.Result()

        def reset(self):
            self._recognizer.Reset()

    def __init__(self, model_path: str, sample_rate: int, **kwargs):
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._vosk_model: VoskModel | None = None
        if not os.path.exists(self._model_path):
            logger.error(f"Vosk model not found at {self._model_path}. STT will not work.")
            # Let it proceed, create_recognizer will fail or main init will handle
        else:
            try:
                self._vosk_model = VoskModel(self._model_path)
                logger.info(f"Vosk model loaded from {self._model_path}. Expected sample rate: {sample_rate}")
            except Exception as e:
                logger.error(f"Failed to load Vosk model from {self._model_path}: {e}", exc_info=True)
                self._vosk_model = None # Ensure it's None on failure

    def create_recognizer(self) -> 'STTEngineBase.RecognizerInstance':
        if not self._vosk_model:
            logger.error("Vosk model not loaded, cannot create recognizer.")
            raise RuntimeError("Vosk STT model is not available.")
        return VoskSTTEngine.VoskRecognizerInstance(self._vosk_model, self._sample_rate)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def delete(self):
        if hasattr(self, '_vosk_model') and self._vosk_model:
            # Vosk Model object doesn't have an explicit delete, relies on GC
            del self._vosk_model
            self._vosk_model = None
            logger.info("Vosk model reference released.")

    async def shutdown(self):
        self.delete()