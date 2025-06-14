import asyncio
import logging
from typing import Optional, Any

from settings import Settings

# Engine implementations
from engines.wake_word.picovoice_porcupine import PicovoicePorcupineEngine
from engines.vad.picovoice_cobra import PicovoiceCobraEngine
from engines.stt.vosk_stt import VoskSTTEngine
from engines.tts.paroli_tts import ParoliTTSEngine
from engines.tts.gemini_tts import GeminiTTSEngine
from engines.tts.applio_tts import ApplioTTSEngine
from engines.tts.hybrid_tts import HybridTTSEngine
from engines.nlu.rasa_nlu import RasaNLUEngine
from engines.llm_logic.langgraph_llm import LangGraphLLMEngine
from engines.llm_logic.ollama_llm import OllamaLLMEngine
from engines.audio_io.sounddevice_io import SoundDeviceInputEngine, SoundDeviceOutputEngine
from engines.communication.mqtt_service import MQTTService

logger = logging.getLogger(__name__)

def create_engine_instance(engine_type: str, engine_name: str, global_settings: Settings) -> Optional[Any]:
    logger.info(f"Creating engine: {engine_name} of type {engine_type}")
    try:
        if engine_type == "wake_word":
            if engine_name == "picovoice_porcupine":
                return PicovoicePorcupineEngine(
                    access_key=global_settings.picovoice.access_key,
                    keyword_paths=global_settings.picovoice.keyword_paths,
                )
        elif engine_type == "vad":
            if engine_name == "picovoice_cobra":
                return PicovoiceCobraEngine(access_key=global_settings.picovoice.access_key)
        elif engine_type == "stt":
            if engine_name == "vosk":
                return VoskSTTEngine(
                    model_path=global_settings.vosk.model_path,
                    sample_rate=global_settings.vosk.sample_rate
                )
        elif engine_type == "tts":
            if engine_name == "paroli":
                return ParoliTTSEngine(config=global_settings.paroli_server)
            elif engine_name == "gemini":
                if not global_settings.gemini_tts.api_key:
                     logger.warning("Gemini TTS engine selected, but API key is missing in settings. It will likely fail.")
                return GeminiTTSEngine(config=global_settings.gemini_tts)
            elif engine_name == "applio":
                if not global_settings.applio_tts.api_url:
                    logger.warning("Applio TTS engine selected, but APPLIO_API_URL is missing in settings. It will likely fail.")
                if not global_settings.applio_tts.pth_path:
                    logger.warning("Applio TTS engine selected, but APPLIO_PTH_PATH is missing in settings. This may be required for voice cloning.")
                return ApplioTTSEngine(config=global_settings.applio_tts)
            elif engine_name == "hybrid":
                online_provider_name = global_settings.engines.tts_online_provider
                offline_provider_name = global_settings.engines.tts_offline_provider
                logger.info(f"Creating Hybrid TTS: Online provider='{online_provider_name}', Offline provider='{offline_provider_name}'")
                online_tts_instance = create_engine_instance("tts", online_provider_name, global_settings)
                offline_tts_instance = create_engine_instance("tts", offline_provider_name, global_settings)
                if not online_tts_instance and not offline_tts_instance:
                    logger.error("Hybrid TTS: Both online and offline providers failed to initialize. Hybrid TTS will not be functional.")
                    return None
                return HybridTTSEngine(config={
                    "online_engine": online_tts_instance,
                    "offline_engine": offline_tts_instance
                })
        elif engine_type == "nlu":
            if engine_name == "rasa":
                return RasaNLUEngine(config=global_settings.rasa_nlu)
        elif engine_type == "llm_logic":
            if engine_name == "langgraph":
                return LangGraphLLMEngine(config={"ai_settings": global_settings.ai})
        elif engine_type == "offline_llm_logic":
             if engine_name == "ollama":
                if global_settings.ollama.base_url and global_settings.ollama.model:
                    return OllamaLLMEngine(config={"ollama_settings": global_settings.ollama})
                else:
                    logger.warning("Ollama engine selected but base_url or model not configured. Skipping.")
                    return None
        elif engine_type == "audio_input":
            if engine_name == "sounddevice":
                return SoundDeviceInputEngine(
                    process_audio_callback=None, # Will be set later
                    loop=asyncio.get_running_loop(),
                    config={
                        "sounddevice_settings": global_settings.sounddevice,
                        "audio_settings": global_settings.audio
                    }
                )
        elif engine_type == "audio_output":
            if engine_name == "sounddevice":
                return SoundDeviceOutputEngine(config={
                     "sounddevice_settings": global_settings.sounddevice,
                     "audio_settings": global_settings.audio
                })
        elif engine_type == "communication":
            if engine_name == "mqtt":
                return MQTTService(config=global_settings.mqtt_broker)
        
        logger.error(f"Unsupported engine name '{engine_name}' for type '{engine_type}'.")
    except Exception as e:
        logger.error(f"Failed to create engine {engine_name} ({engine_type}): {e}", exc_info=True)
    return None