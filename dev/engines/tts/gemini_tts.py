import asyncio
import logging
from typing import AsyncIterator, Dict, Any, Optional
# from itertools import tee # Больше не нужен tee

from .base import TTSEngineBase
from settings import GeminiTTSSettings # Specific settings for Gemini TTS

# Attempt to import Google GenAI
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None # type: ignore
    types = None # type: ignore

logger = logging.getLogger(__name__)

class GeminiTTSEngine(TTSEngineBase):
    def __init__(self, config: GeminiTTSSettings): 
        self.settings: GeminiTTSSettings = config
        self.client: Optional[genai.Client] = None 

        if not GENAI_AVAILABLE:
            logger.error("GeminiTTSEngine: 'google-generativeai' library not found. Please install it.")
        if not self.settings.api_key:
            logger.warning("GeminiTTSEngine: API key not configured. TTS will not work.")

    async def startup(self):
        if not GENAI_AVAILABLE:
            logger.error("GeminiTTSEngine: Cannot start, 'google-generativeai' is not available.")
            return
        if not self.settings.api_key:
            logger.error("GeminiTTSEngine: Cannot start, API key is missing.")
            return
        
        try:
            if genai and types: 
                 self.client = genai.Client(api_key=self.settings.api_key)
                 logger.info(f"GeminiTTSEngine: Client initialized. Will use model '{self.settings.model}' for synthesis.")
            else:
                logger.error("GeminiTTSEngine: GenAI library or types not available. Client not initialized.")
                self.client = None 
        except Exception as e:
            logger.error(f"GeminiTTSEngine: Failed to initialize GenAI client: {e}", exc_info=True)
            self.client = None

    async def shutdown(self):
        self.client = None 
        logger.info("GeminiTTSEngine shutdown.")

    async def is_healthy(self) -> bool:
        if not GENAI_AVAILABLE:
            return False
        if not self.settings.api_key:
            logger.warning("GeminiTTSEngine is unhealthy: API key not configured.")
            return False
        if not self.client: 
            logger.warning("GeminiTTSEngine is unhealthy: Client not initialized (startup may have failed or library missing).")
            return False
        return True

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        if not await self.is_healthy():
            logger.error("GeminiTTSEngine not healthy or not configured. Cannot synthesize TTS.")
            return # Пустой async generator

        if not types or not genai: 
            logger.error("GeminiTTSEngine: 'google.generativeai' or 'types' not available. Cannot synthesize.")
            return # Пустой async generator

        try:
            loop = asyncio.get_running_loop()

            tts_generation_config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.settings.voice_name,
                        )
                    )
                )
            )
            
            def sync_call_generate_content() -> Optional[genai.types.GenerateContentResponse]:
                if not self.client: 
                    raise RuntimeError("Gemini client not available in sync_call_generate_content")
                # Для TTS с genai.Client, generate_content возвращает один GenerateContentResponse
                response = self.client.models.generate_content(
                    model=self.settings.model, 
                    contents=text,
                    config=tts_generation_config 
                )
                if isinstance(response, genai.types.GenerateContentResponse):
                    return response
                # Если API вернул кортеж, где первый элемент - это GenerateContentResponse
                elif isinstance(response, tuple) and len(response) > 0 and isinstance(response[0], genai.types.GenerateContentResponse):
                    logger.debug("GeminiTTS: API returned a tuple, using its first element as GenerateContentResponse.")
                    return response[0]
                else:
                    logger.error(f"GeminiTTS: Unexpected API response type: {type(response)}. Expected GenerateContentResponse or tuple containing it.")
                    return None

            logger.info(f"GeminiTTS: Synthesizing for: '{text[:30]}...' using model '{self.settings.model}' and voice '{self.settings.voice_name}'")
            
            api_response: Optional[genai.types.GenerateContentResponse] = await loop.run_in_executor(None, sync_call_generate_content)
            
            if api_response:
                logger.debug(f"GeminiTTS: Received API response of type {type(api_response)}.")
                if api_response.candidates and \
                   len(api_response.candidates) > 0 and \
                   api_response.candidates[0].content and \
                   api_response.candidates[0].content.parts and \
                   len(api_response.candidates[0].content.parts) > 0:
                    
                    audio_part = api_response.candidates[0].content.parts[0]
                    if audio_part.inline_data and audio_part.inline_data.data:
                        logger.info(f"GeminiTTS: Yielding {len(audio_part.inline_data.data)} bytes of audio data.")
                        yield audio_part.inline_data.data
                    else:
                        logger.warning("GeminiTTS: Audio part found but inline_data.data is missing or empty.")
                else:
                    logger.warning("GeminiTTS: API response did not contain the expected candidate/content/part structure for audio.")
            else:
                logger.error("GeminiTTS: No valid API response received from sync_call_generate_content.")

        except Exception as e:
            logger.error(f"GeminiTTS: Error during speech synthesis: {e}", exc_info=True)
            # Чтобы сделать его async generator даже при ошибке до yield
            # if False: yield b"" 
            # Однако, если return используется в async gen, он вызывает StopAsyncIteration, что нормально.

    async def synthesize_once(self, text: str) -> bytes:
        if not await self.is_healthy():
            logger.error("GeminiTTSEngine not healthy. Cannot synthesize TTS.")
            return b""
            
        all_audio = bytearray()
        try:
            # synthesize_stream теперь должен yield один раз (или ни разу при ошибке)
            async for chunk in self.synthesize_stream(text):
                all_audio.extend(chunk)
            
            if not all_audio:
                logger.warning("GeminiTTS: synthesize_once resulted in empty audio.")
            else:
                logger.info(f"GeminiTTS: synthesize_once collected {len(all_audio)} bytes.")
            return bytes(all_audio)
        except Exception as e:
            logger.error(f"GeminiTTS: Error during synthesize_once: {e}", exc_info=True)
            return b""

    def get_output_sample_rate(self) -> int:
        return self.settings.sample_rate