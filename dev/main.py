import asyncio
import base64
import json
import logging
import os
import struct
import wave # Added for loading activation sound
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# Project specific imports
from settings import settings, Settings, AudioSettings # Import the main Settings class and AudioSettings for type hint
# MODIFIED: Import new functions from connectivity
from connectivity import is_internet_available, start_connectivity_monitoring, stop_connectivity_monitoring 

from tools.scheduler import init_db as init_scheduler_db
from utils.music_db import init_music_likes_table
from services.reminder_checker import start_reminder_checker, stop_reminder_checker

# Engine imports
from engines.wake_word.base import WakeWordEngineBase
from engines.vad.base import VADEngineBase
from engines.stt.base import STTEngineBase, STTEngineBase as STTRecognizerProvider # Alias for clarity
from engines.tts.base import TTSEngineBase
from engines.nlu.base import NLUEngineBase
from engines.llm_logic.base import LLMLogicEngineBase
from engines.audio_io.input_base import AudioInputEngineBase, ProcessAudioCallback
from engines.audio_io.output_base import AudioOutputEngineBase
from engines.communication.base import CommunicationServiceBase

# Engine factory
from engines.wake_word.picovoice_porcupine import PicovoicePorcupineEngine
from engines.vad.picovoice_cobra import PicovoiceCobraEngine
from engines.stt.vosk_stt import VoskSTTEngine
from engines.tts.paroli_tts import ParoliTTSEngine
from engines.tts.gemini_tts import GeminiTTSEngine
from engines.tts.hybrid_tts import HybridTTSEngine
from engines.nlu.rasa_nlu import RasaNLUEngine
from engines.llm_logic.langgraph_llm import LangGraphLLMEngine
from engines.llm_logic.ollama_llm import OllamaLLMEngine
from engines.audio_io.sounddevice_io import SoundDeviceInputEngine, SoundDeviceOutputEngine
from engines.communication.mqtt_service import MQTTService
from engines.offline_processing.base import OfflineCommandProcessorBase
from engines.offline_processing.default_processor import DefaultOfflineCommandProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

wake_word_engine: Optional[WakeWordEngineBase] = None
vad_engine: Optional[VADEngineBase] = None
stt_engine: Optional[STTRecognizerProvider] = None
tts_engine: Optional[TTSEngineBase] = None
nlu_engine: Optional[NLUEngineBase] = None
llm_logic_engine: Optional[LLMLogicEngineBase] = None
offline_llm_logic_engine: Optional[LLMLogicEngineBase] = None
audio_input_engine: Optional[AudioInputEngineBase] = None
audio_output_engine: Optional[AudioOutputEngineBase] = None
comm_service: Optional[CommunicationServiceBase] = None
offline_command_processor: Optional[OfflineCommandProcessorBase] = None

class ConnectionManager: pass
manager: Optional[ConnectionManager] = None

ASSISTANT_THREAD_ID = "lumi-voice-assistant-session"

def create_engine_instance(engine_type: str, engine_name: str, global_settings: Settings) -> Optional[Any]:
    logger.info(f"Creating engine: {engine_name} of type {engine_type}")
    try:
        if engine_type == "wake_word":
            if engine_name == "picovoice_porcupine":
                return PicovoicePorcupineEngine(
                    access_key=global_settings.picovoice.access_key,
                    keywords=global_settings.picovoice.porcupine_keywords
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
            elif engine_name == "hybrid":
                online_provider_name = global_settings.engines.tts_online_provider
                offline_provider_name = global_settings.engines.tts_offline_provider
                logger.info(f"Creating Hybrid TTS: Online provider='{online_provider_name}', Offline provider='{offline_provider_name}'")
                online_tts_instance = create_engine_instance("tts", online_provider_name, global_settings)
                offline_tts_instance = create_engine_instance("tts", offline_provider_name, global_settings)
                if not online_tts_instance:
                    logger.warning(f"Hybrid TTS: Online provider '{online_provider_name}' failed to initialize.")
                if not offline_tts_instance:
                    logger.warning(f"Hybrid TTS: Offline provider '{offline_provider_name}' failed to initialize.")
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
                return LangGraphLLMEngine(config={
                    "ai_settings": global_settings.ai,
                    "postgres_settings": global_settings.postgres
                })
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

async def initialize_global_engines(app_settings: Settings):
    global wake_word_engine, vad_engine, stt_engine, tts_engine, nlu_engine
    global llm_logic_engine, offline_llm_logic_engine
    global audio_input_engine, audio_output_engine, comm_service, manager
    global offline_command_processor

    logger.info("Initializing global engines...")

    comm_service = create_engine_instance("communication", app_settings.engines.communication_engine, app_settings)
    if comm_service:
        await comm_service.startup()
        from tools.device_control import initialize_device_control_tool
        initialize_device_control_tool(comm_service)
    else:
        logger.error("Communication service failed to initialize. Dependent services will be affected.")

    if comm_service:
        offline_command_processor = DefaultOfflineCommandProcessor(comm_service=comm_service)
        logger.info("DefaultOfflineCommandProcessor initialized.")
    else:
        offline_command_processor = None
        logger.warning("OfflineCommandProcessor not initialized as comm_service is unavailable.")

    wake_word_engine = create_engine_instance("wake_word", app_settings.engines.wake_word_engine, app_settings)
    vad_engine = create_engine_instance("vad", app_settings.engines.vad_engine, app_settings)
    stt_engine = create_engine_instance("stt", app_settings.engines.stt_engine, app_settings)
    tts_engine = create_engine_instance("tts", app_settings.engines.tts_engine, app_settings)
    nlu_engine = create_engine_instance("nlu", app_settings.engines.nlu_engine, app_settings)
    llm_logic_engine = create_engine_instance("llm_logic", app_settings.engines.llm_logic_engine, app_settings)
    offline_llm_logic_engine = create_engine_instance("offline_llm_logic", app_settings.engines.offline_llm_engine, app_settings)
    
    if not all([wake_word_engine, vad_engine, stt_engine, tts_engine, nlu_engine, comm_service]):
        logger.critical("One or more core processing or communication engines failed to initialize. Functionality will be severely limited.")
    if not tts_engine:
        logger.critical("TTS Engine (potentially Hybrid) failed to initialize. TTS functionality will be unavailable.")
    if not llm_logic_engine:
        logger.warning("Online LLM logic engine failed to initialize. Online general queries will not work.")
    if not offline_llm_logic_engine:
        logger.warning("Offline LLM logic engine failed to initialize. Offline general queries will not work.")
    if not offline_command_processor:
        logger.warning("Offline command processor is not available. Offline commands might not work as expected.")
    
    manager = ConnectionManager(
        wake_word_engine=wake_word_engine,
        vad_engine=vad_engine,
        stt_provider=stt_engine,
        tts_engine=tts_engine,
        nlu_engine=nlu_engine,
        llm_logic_engine=llm_logic_engine,
        offline_llm_logic_engine=offline_llm_logic_engine,
        comm_service=comm_service,
        offline_processor=offline_command_processor,
        global_audio_settings=app_settings.audio,
        vad_processing_settings=app_settings.vad_config
    )
    logger.info("ConnectionManager initialized.")

    _audio_input_engine_instance = create_engine_instance("audio_input", app_settings.engines.audio_input_engine, app_settings)
    if _audio_input_engine_instance and isinstance(_audio_input_engine_instance, AudioInputEngineBase):
        async def process_audio_wrapper(audio_chunk: bytes):
            if manager:
                await manager.process_audio(audio_chunk, ASSISTANT_THREAD_ID, is_local_source=True)
        _audio_input_engine_instance._process_audio_cb = process_audio_wrapper # type: ignore
        audio_input_engine = _audio_input_engine_instance
        logger.info(f"AudioInputEngine ({app_settings.engines.audio_input_engine}) created.")
        if audio_input_engine and audio_input_engine.is_enabled:
            if wake_word_engine and audio_input_engine.sample_rate != wake_word_engine.sample_rate:
                 logger.warning(f"AudioInput sample rate ({audio_input_engine.sample_rate}) "
                                f"differs from WakeWord sample rate ({wake_word_engine.sample_rate}).")
            if wake_word_engine and audio_input_engine.frame_length != wake_word_engine.frame_length:
                 logger.warning(f"AudioInput frame length ({audio_input_engine.frame_length}) "
                                f"differs from WakeWord frame length ({wake_word_engine.frame_length}). Mismatched audio chunks.")
        else:
            logger.info(f"AudioInputEngine ({app_settings.engines.audio_input_engine}) is disabled or failed to initialize.")
    else:
        logger.warning(f"Could not create or wrong type for AudioInputEngine: {app_settings.engines.audio_input_engine}")

    audio_output_engine = create_engine_instance("audio_output", app_settings.engines.audio_output_engine, app_settings)
    if audio_output_engine and audio_output_engine.is_enabled:
        logger.info(f"AudioOutputEngine ({app_settings.engines.audio_output_engine}) created and enabled.")
        if manager: manager.set_local_audio_output(audio_output_engine)
    else:
        logger.info(f"AudioOutputEngine ({app_settings.engines.audio_output_engine}) is disabled or failed to initialize.")

    if tts_engine: await tts_engine.startup()
    if llm_logic_engine: await llm_logic_engine.startup()
    if offline_llm_logic_engine: await offline_llm_logic_engine.startup()

    logger.info("Global engines initialization phase complete.")


async def shutdown_global_engines():
    logger.info("Shutting down global engines...")
    engine_list = [
        tts_engine, llm_logic_engine, offline_llm_logic_engine, nlu_engine,
        wake_word_engine, vad_engine, stt_engine, audio_input_engine, 
        audio_output_engine, comm_service
    ]
    for engine in engine_list:
        if engine and hasattr(engine, "shutdown"):
            try:
                logger.info(f"Shutting down {engine.__class__.__name__}...")
                await engine.shutdown()
                logger.info(f"{engine.__class__.__name__} shut down successfully.")
            except Exception as e:
                logger.error(f"Error shutting down {engine.__class__.__name__}: {e}", exc_info=True)
    logger.info("Global engines shutdown phase complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Lifespan: Startup phase beginning...")
    await start_connectivity_monitoring() # MODIFIED
    try:
        init_scheduler_db()
        init_music_likes_table()
    except Exception as e:
        logger.error(f"Failed to initialize scheduler or music database: {e}. Some features might not work.", exc_info=True)

    await initialize_global_engines(settings)

    global tts_engine, audio_output_engine # manager is already global
    if tts_engine and audio_output_engine and await tts_engine.is_healthy() and audio_output_engine.is_enabled:
        if not settings.scheduler_db_path:
            logger.warning("Scheduler DB path not set. Reminder checker might not function correctly.")
        await start_reminder_checker(tts_engine, audio_output_engine)
    else:
        logger.warning("TTS or Audio Output engine not available/healthy, reminder checker not started.")

    if audio_input_engine and audio_input_engine.is_enabled and \
       audio_output_engine and audio_output_engine.is_enabled and \
       manager and not manager.is_websocket_active:
        logger.info("Lifespan: No active WebSocket, starting local audio I/O.")
        audio_output_engine.start()
        audio_input_engine.start()
        manager.state = "wakeword"
        logger.info("Lifespan: Local audio interface activated, waiting for wake word.")
    
    logger.info("Lifespan: Startup phase complete.")
    yield
    logger.info("Lifespan: Shutdown phase beginning...")
    await stop_reminder_checker()
    await shutdown_global_engines()
    await stop_connectivity_monitoring() # MODIFIED
    logger.info("Lifespan: Shutdown phase complete.")


app = FastAPI(
    title="Lumi Voice Assistant Backend (Modular)",
    lifespan=lifespan
)

# Ensure the 'static' directory exists at the root of your project or adjust path
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
    logger.info(f"Created static directory at: {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_root():
    static_file_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(static_file_path):
        # Fallback or more informative error
        logger.error(f"Frontend file not found at: {static_file_path}")
        return HTTPException(status_code=404, detail=f"Frontend file 'index.html' not found in '{static_dir}'. Please ensure it exists.")
    from fastapi.responses import FileResponse
    return FileResponse(static_file_path)


class ConnectionManager:
    def __init__(self, wake_word_engine: Optional[WakeWordEngineBase],
                 vad_engine: Optional[VADEngineBase],
                 stt_provider: Optional[STTRecognizerProvider],
                 tts_engine: Optional[TTSEngineBase],
                 nlu_engine: Optional[NLUEngineBase],
                 llm_logic_engine: Optional[LLMLogicEngineBase],
                 offline_llm_logic_engine: Optional[LLMLogicEngineBase],
                 comm_service: Optional[CommunicationServiceBase],
                 offline_processor: Optional[OfflineCommandProcessorBase],
                 global_audio_settings: AudioSettings,
                 vad_processing_settings: "settings.VADSettings"): # type: ignore

        self.wake_word_engine = wake_word_engine
        self.vad_engine = vad_engine
        self.stt_provider = stt_provider
        self.tts_engine = tts_engine
        self.nlu_engine = nlu_engine
        self.llm_logic_engine = llm_logic_engine
        self.offline_llm_logic_engine = offline_llm_logic_engine
        self.comm_service = comm_service
        self.offline_processor = offline_processor
        self._last_mentioned_device_for_pronoun: Optional[str] = None
        self.global_audio_settings = global_audio_settings
        self.vad_processing_settings = vad_processing_settings

        self.websocket: Optional[WebSocket] = None
        self.stt_recognizer: Optional[STTEngineBase.RecognizerInstance] = None
        self.state: str = "disconnected"
        self.audio_buffer: bytearray = bytearray()
        self.silence_frames_count: int = 0
        self.frames_in_listening: int = 0
        self.llm_tts_task: Optional[asyncio.Task] = None
        self.is_websocket_active: bool = False
        self.local_audio_output_engine: Optional[AudioOutputEngineBase] = None

        self.estimated_noise_floor: float = self.vad_processing_settings.probability_threshold
        self.current_dynamic_vad_threshold: float = self.vad_processing_settings.probability_threshold
        self.potential_silence_frames: int = 0
        self.was_recently_voiced: bool = False

        self.activation_sound_bytes: Optional[bytes] = None
        self.activation_sound_sample_rate: Optional[int] = None
        self.activation_sound_channels: Optional[int] = None
        self.activation_sound_sampwidth: Optional[int] = None

        try:
            if self.global_audio_settings.play_activation_sound and self.global_audio_settings.activation_sound_path:
                sound_path = self.global_audio_settings.activation_sound_path
                if sound_path and os.path.exists(sound_path): # Check if sound_path is not None
                    with wave.open(sound_path, 'rb') as wf:
                        self.activation_sound_bytes = wf.readframes(wf.getnframes())
                        self.activation_sound_sample_rate = wf.getframerate()
                        self.activation_sound_channels = wf.getnchannels()
                        self.activation_sound_sampwidth = wf.getsampwidth()
                        logger.info(f"Activation sound loaded: {sound_path}, SR={self.activation_sound_sample_rate}Hz, Channels={self.activation_sound_channels}, Width={self.activation_sound_sampwidth}bytes, Size={len(self.activation_sound_bytes)} bytes")
                        if self.activation_sound_channels != 1:
                            logger.warning(f"Activation sound '{sound_path}' is not mono. Playback might be suboptimal.")
                        if self.activation_sound_sampwidth != 2:
                            logger.warning(f"Activation sound '{sound_path}' sample width is {self.activation_sound_sampwidth} bytes (expected 2 for int16). Playback might be incorrect.")
                elif sound_path: # sound_path was set but file doesn't exist
                     logger.warning(f"Activation sound file not found: {sound_path}. Activation sound will not be played.")
                else: # activation_sound_path was None
                    logger.warning("Activation sound path is None. Activation sound will not be played.")
            elif self.global_audio_settings.play_activation_sound:
                logger.warning("Activation sound is enabled but no path is specified or found (ACTIVATION_SOUND_PATH).")
        except Exception as e:
            logger.error(f"Failed to load activation sound: {e}", exc_info=True)
            self.activation_sound_bytes = None # Ensure it's None on failure

        if self.wake_word_engine and self.global_audio_settings.frame_length != self.wake_word_engine.frame_length:
            logger.warning(f"Global audio frame length ({self.global_audio_settings.frame_length}) differs from WakeWord frame length ({self.wake_word_engine.frame_length}).")
        if self.vad_engine and self.global_audio_settings.frame_length != self.vad_engine.frame_length:
            logger.warning(f"Global audio frame length ({self.global_audio_settings.frame_length}) differs from VAD frame length ({self.vad_engine.frame_length}).")
        if self.stt_provider:
            try:
                self.stt_recognizer = self.stt_provider.create_recognizer()
                logger.info("ConnectionManager: STT recognizer instance created during init.")
            except Exception as e:
                logger.error(f"ConnectionManager: Failed to create STT recognizer during init: {e}", exc_info=True)
        else:
            logger.warning("ConnectionManager: STT provider not available during init. STT will not work.")

    def set_local_audio_output(self, local_audio_out_engine: AudioOutputEngineBase):
        self.local_audio_output_engine = local_audio_out_engine

    async def connect(self, websocket: WebSocket):
        global audio_input_engine, audio_output_engine
        if self.websocket is not None and self.websocket.client_state == WebSocketState.CONNECTED:
            logger.warning("Another WebSocket client tried to connect while one is active. Rejecting.")
            await websocket.accept() # Must accept before closing with a reason
            await websocket.close(code=1008, reason="Server busy. Another client is connected.")
            return False

        await websocket.accept()
        self.websocket = websocket
        self.state = "wakeword"
        self.audio_buffer = bytearray()
        self.silence_frames_count = 0
        self.frames_in_listening = 0
        self.llm_tts_task = None
        self.is_websocket_active = True

        if audio_input_engine and audio_input_engine.is_enabled:
            logger.info("WebSocket connected, pausing local audio input.")
            audio_input_engine.pause()
        if audio_output_engine and audio_output_engine.is_enabled:
            logger.info("WebSocket connected, stopping any local audio output queue.")
            audio_output_engine.stop() # Stop to clear queue and ensure it doesn't interfere

        if not self.stt_provider:
            logger.error("STT provider not available!")
            await self.send_error("Server STT error.")
            await self.disconnect(code=1011, reason="Server STT error") # Pass reason
            return False
        
        try:
            if not self.stt_recognizer: # Should have been created in __init__
                logger.warning("STT recognizer was not created in __init__, attempting now.")
                self.stt_recognizer = self.stt_provider.create_recognizer()
                logger.info("STT recognizer created on connect.")
            else:
                self.stt_recognizer.reset()
                logger.info("STT recognizer reset on connect.")
        except Exception as e:
            logger.error(f"Failed to create/reset STT recognizer: {e}", exc_info=True)
            await self.send_error("Server STT engine failure.")
            await self.disconnect(code=1011, reason="Server STT engine failure")
            return False
        
        if not self.tts_engine or not await self.tts_engine.is_healthy():
            logger.warning("TTS engine not available or not healthy! TTS might not work.")
            await self.send_status("tts_warning", "TTS engine might be unavailable.")


        logger.info("Client connected via WebSocket.")
        await self.send_status("wakeword_listening", "Waiting for wake word...")
        return True

    async def disconnect(self, code: int = 1000, reason: str = "Client disconnected"):
        global audio_input_engine, audio_output_engine
        self.is_websocket_active = False # Set this first

        if self.llm_tts_task and not self.llm_tts_task.done():
            self.llm_tts_task.cancel()
            try: await self.llm_tts_task
            except asyncio.CancelledError: logger.info("LLM/TTS task cancelled on disconnect.")
            except Exception as e: logger.error(f"Error during LLM/TTS task cancellation on disconnect: {e}")
        self.llm_tts_task = None

        if self.websocket:
            ws_temp = self.websocket
            self.websocket = None # Nullify before trying to close
            if ws_temp.client_state == WebSocketState.CONNECTED:
                try:
                    logger.info(f"Closing WebSocket with code {code} and reason '{reason}'")
                    await ws_temp.close(code=code, reason=reason)
                except Exception as e:
                    logger.warning(f"Error closing WebSocket: {e}")
            elif ws_temp.client_state != WebSocketState.DISCONNECTED:
                 logger.info(f"WebSocket was in state {ws_temp.client_state}, not attempting close.")
        
        current_state = self.state
        self.state = "disconnected" # Final state after cleanup

        logger.info(f"WebSocket client disconnected (previous state: {current_state}). Code: {code}, Reason: {reason}")

        if audio_input_engine and audio_input_engine.is_enabled:
            logger.info("WebSocket disconnected, resuming local audio input.")
            if not audio_input_engine.is_running: audio_input_engine.start()
            audio_input_engine.resume()
            self.state = "wakeword" # Transition back to wakeword for local mode
            logger.info("Local audio interface active, waiting for wake word.")
        if audio_output_engine and audio_output_engine.is_enabled:
            if not audio_output_engine.is_running: audio_output_engine.start()

    async def _send_json(self, data: dict):
        if self.is_websocket_active and self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await self.websocket.send_json(data)
                return True
            except WebSocketDisconnect:
                logger.warning("WebSocket disconnected while trying to send JSON. Client might have closed connection.")
                await self.disconnect(reason="WebSocket disconnected during send")
                return False
            except Exception as e:
                 logger.error(f"Error sending JSON to client: {e}")
                 return False
        return False

    async def send_status(self, status_code: str, message: str):
        await self._send_json({"type": "status", "code": status_code, "message": message})

    async def send_transcript(self, transcript: str, is_final: bool):
         await self._send_json({"type": "transcript", "text": transcript, "is_final": is_final})

    async def send_tts_info(self, sample_rate: int, channels: int = 1, bit_depth: int = 16): # Added
        if self.is_websocket_active:
            await self._send_json({
                "type": "tts_info",
                "sample_rate": sample_rate,
                "channels": channels,
                "bit_depth": bit_depth # Assuming 16-bit PCM for now
            })

    async def send_tts_chunk(self, audio_chunk_b64: str):
        if self.is_websocket_active:
            await self._send_json({"type": "tts_chunk", "data": audio_chunk_b64})

    async def send_tts_finished(self):
        if self.is_websocket_active:
            await self._send_json({"type": "tts_finished"})

    async def send_error(self, error_message: str):
        logger.error(f"Sending error to client: {error_message}")
        if self.is_websocket_active: # Check active before sending
            await self._send_json({"type": "error", "message": error_message})
    
    async def _run_llm_tts_or_offline(self, text: str, thread_id: str):
            response_text = "" # Initialize response_text
            tts_audio_bytes_for_local = bytearray()
            current_tts_sample_rate_for_playback = None 

            try:
                if self.is_websocket_active:
                    await self.send_status("processing_started", "Thinking...")
                else:
                    logger.info("Processing request (local audio)...")
                self.state = "processing"

                online_capable = await is_internet_available()
                # Determine if an online LLM attempt should be made
                attempt_online_llm = online_capable and settings.ai.online_mode and self.llm_logic_engine
                
                online_llm_attempt_made = False
                online_llm_succeeded = False

                if attempt_online_llm:
                    online_llm_attempt_made = True
                    logger.info("Using online LLM (LangGraph).")
                    try:
                        lumi_response = await self.llm_logic_engine.ask(text, thread_id=thread_id) # type: ignore
                        _candidate_response = lumi_response if isinstance(lumi_response, str) else lumi_response.content # type: ignore
                        
                        # Check if the response is not empty and not a generic error message from the LLM itself
                        # Error messages from LangGraphLLM.ask start with "Sorry, an error occurred" or "Sorry, the language model is not available"
                        if _candidate_response and not (
                            _candidate_response.startswith("Sorry, an error occurred") or \
                            _candidate_response.startswith("Sorry, the language model is not available")
                            ):
                            response_text = _candidate_response
                            online_llm_succeeded = True
                            logger.info("Online LLM successfully provided a response.")
                        else:
                            logger.warning(f"Online LLM returned an empty or error-like response: '{_candidate_response}'. Will fall back to offline.")
                            # response_text remains empty, online_llm_succeeded is false
                    except Exception as e:
                        logger.error(f"Exception during online LLM call: {e}. Will fall back to offline.", exc_info=True)
                        # response_text remains empty, online_llm_succeeded is false
                
                # If online processing was not attempted, or if it was attempted but did not succeed
                if not online_llm_succeeded:
                    if online_llm_attempt_made:
                        logger.info("Online LLM attempt did not yield a usable response. Proceeding with offline processing.")
                    else: # Online LLM was not attempted
                        logger.info(f"Offline processing path activated (Online LLM not attempted. Internet: {online_capable}, Online Mode: {settings.ai.online_mode}).")

                    nlu_result = None
                    if self.nlu_engine:
                        logger.debug("Attempting NLU parse for potential offline command.")
                        nlu_result = await self.nlu_engine.parse(text)

                    if nlu_result and nlu_result.get("intent"):
                        if self.offline_processor:
                            logger.info(f"Offline NLU parsed command: {nlu_result}")
                            resolved_command = await self.offline_processor.process_nlu_result(
                                nlu_result, self._last_mentioned_device_for_pronoun
                            )
                            response_text = await self.offline_processor.execute_resolved_command(resolved_command)
                            if resolved_command.get("executable") and \
                            resolved_command.get("resolved_device_name_for_context_update"):
                                self._last_mentioned_device_for_pronoun = resolved_command["resolved_device_name_for_context_update"]
                                logger.debug(f"ConnectionManager: Updated pronoun context to '{self._last_mentioned_device_for_pronoun}'")
                        else:
                            logger.warning("Offline NLU parsed a command, but no offline_processor is available.")
                            response_text = "I understood an offline command, but cannot process it right now."
                    else: # NLU did not identify a command / NLU unavailable
                        if self.offline_llm_logic_engine:
                            logger.info("NLU did not identify a command / NLU unavailable. Trying offline LLM (Ollama).")
                            lumi_response_offline = await self.offline_llm_logic_engine.ask(text, thread_id=thread_id)
                            _candidate_offline_response = lumi_response_offline if isinstance(lumi_response_offline, str) else lumi_response_offline.content # type: ignore
                            
                            # Check if offline LLM response is not empty and not its own generic error
                            # Error messages from OllamaLLMEngine.ask start with "Sorry, an error occurred" or "Sorry, the offline language model is not available"
                            if _candidate_offline_response and not (
                                _candidate_offline_response.startswith("Sorry, an error occurred") or \
                                _candidate_offline_response.startswith("Sorry, the offline language model is not available")
                                ):
                                response_text = _candidate_offline_response
                            else:
                                logger.warning(f"Offline LLM also returned an empty or error-like response: '{_candidate_offline_response}'")
                                if not response_text: # Ensure we set a message if it's still empty after this path
                                     response_text = "The offline assistant didn't provide a response."
                        else: # No NLU match, and no offline LLM
                            logger.info("NLU did not identify a command, and offline LLM is not available.")
                            if not response_text: # Ensure we set a message if it's still empty
                                if online_llm_attempt_made : # Online was tried and failed, now offline options also exhausted
                                    response_text = "I couldn't process this online, and no suitable offline action was found."
                                else: # Online was not even an option, and offline options exhausted
                                    response_text = "I can only process specific commands offline, and this wasn't one of them. The general offline assistant is also unavailable."
                
                # Final fallback if response_text is STILL empty after all attempts (should be rare)
                if not response_text:
                    logger.error("All processing paths (online, NLU, offline LLM) failed to produce a response_text.")
                    response_text = "I'm sorry, I was unable to process your request at this time."
                
                # --- TTS Section ---
                if response_text and self.tts_engine:
                    active_tts_synthesizer = None
                    if isinstance(self.tts_engine, HybridTTSEngine):
                        active_tts_synthesizer = await self.tts_engine.get_active_engine_for_synthesis()
                    elif hasattr(self.tts_engine, 'get_output_sample_rate'): 
                        active_tts_synthesizer = self.tts_engine
                    
                    if active_tts_synthesizer and await active_tts_synthesizer.is_healthy():
                        current_tts_sample_rate_for_playback = active_tts_synthesizer.get_output_sample_rate()
                        logger.info(f"TTS: Using {active_tts_synthesizer.__class__.__name__} with SR {current_tts_sample_rate_for_playback}Hz.")

                        if self.is_websocket_active and current_tts_sample_rate_for_playback:
                            await self.send_tts_info(sample_rate=current_tts_sample_rate_for_playback)

                        if self.is_websocket_active:
                            await self.send_status("speaking_started", "Speaking...")
                        else:
                            logger.info(f"Synthesizing for local playback (SR: {current_tts_sample_rate_for_playback}Hz): '{response_text[:50]}...'")
                        self.state = "speaking"

                        async for tts_chunk in active_tts_synthesizer.synthesize_stream(response_text):
                            if self.is_websocket_active:
                                await self.send_tts_chunk(base64.b64encode(tts_chunk).decode('utf-8'))
                            if self.local_audio_output_engine and self.local_audio_output_engine.is_enabled:
                                tts_audio_bytes_for_local.extend(tts_chunk)
                        
                        if self.is_websocket_active: await self.send_tts_finished()
                        
                        if tts_audio_bytes_for_local and self.local_audio_output_engine and \
                        self.local_audio_output_engine.is_enabled and not self.is_websocket_active:
                            logger.info(f"Queueing {len(tts_audio_bytes_for_local)} bytes for local TTS (SR: {current_tts_sample_rate_for_playback}Hz).")
                            self.local_audio_output_engine.play_tts_bytes(
                                bytes(tts_audio_bytes_for_local),
                                sample_rate=current_tts_sample_rate_for_playback
                            )
                    else:
                        logger.error("TTS: No active and healthy synthesizer found or TTS engine is not correctly configured.")
                        if self.is_websocket_active: await self.send_error("Sorry, I can't speak right now (TTS synth error).")
                        # If TTS fails, response_text still holds the text, but it won't be spoken.
                        # This is acceptable; the user might see it if there's a UI.

                elif not response_text: # Should be caught by the final fallback above, but as a safeguard
                    logger.warning("No response text generated by any LLM/NLU path after all fallbacks.")
                    if self.is_websocket_active:
                        await self.send_error("Sorry, I could not process your request.")
                elif not self.tts_engine:
                    logger.error("TTS engine not configured, cannot speak response.")
                    if self.is_websocket_active:
                        await self.send_error("Sorry, I can't speak right now (TTS not configured).")

            except asyncio.CancelledError:
                logger.info("LLM/TTS/Offline background task cancelled.")
            except Exception as e:
                logger.error(f"Error in _run_llm_tts_or_offline: {e}", exc_info=True)
                if self.is_websocket_active: 
                    await self.send_error(f"Error processing request: {str(e)}")
            finally:
                if self.state != "disconnected": 
                    self.state = "wakeword"
                    if self.is_websocket_active:
                        await self.send_status("wakeword_listening", "Waiting for wake word...")
                    else:
                        logger.info("Local: Waiting for wake word...")
                self.llm_tts_task = None


    async def process_audio(self, audio_chunk: bytes, thread_id: str, is_local_source: bool = False):
        if self.state in ["processing", "speaking"]: return

        if not all([self.wake_word_engine, self.vad_engine, self.stt_provider, self.stt_recognizer]):
            logger.error("Core audio processing engines not ready in ConnectionManager.")
            if not is_local_source and self.is_websocket_active : await self.send_error("Server audio engines not ready.")
            return

        bytes_per_sample = 2 # Assuming 16-bit audio
        expected_bytes = self.global_audio_settings.frame_length * bytes_per_sample

        if len(audio_chunk) != expected_bytes:
            # This can happen with AudioWorklet if block sizes vary, though it tries to send fixed chunks.
            # For local SoundDevice, it should always match.
            logger.debug(f"Audio chunk size mismatch: got {len(audio_chunk)}, expected {expected_bytes}. Buffering or adjusting might be needed if persistent.")
            # For now, we'll skip if it's not the exact size expected by WW/VAD.
            # More robust handling might involve an internal buffer in ConnectionManager
            # to re-chunk audio to the correct frame_length.
            return

        try:
            pcm = struct.unpack_from(f"{self.global_audio_settings.frame_length}h", audio_chunk)
        except struct.error as e:
            logger.error(f"Audio chunk unpack error: {e}. Length: {len(audio_chunk)}")
            return

        if self.state == "wakeword":
            keyword_index = self.wake_word_engine.process(pcm)
            if keyword_index >= 0:
                logger.info("Wake word detected!")

                if self.global_audio_settings.play_activation_sound:
                    if self.is_websocket_active:
                        logger.info("Sending cue to client to play activation sound.")
                        await self.send_status("play_activation_sound_cue", "Client should play activation sound.")
                    elif self.activation_sound_bytes and self.activation_sound_sample_rate and \
                         self.local_audio_output_engine and self.local_audio_output_engine.is_enabled:
                        logger.info(f"Playing activation sound locally (SR: {self.activation_sound_sample_rate}Hz).")
                        self.local_audio_output_engine.play_tts_bytes(
                            self.activation_sound_bytes,
                            sample_rate=self.activation_sound_sample_rate
                        )
                
                self.state = "listening"
                self.audio_buffer.clear()
                self.silence_frames_count = 0
                self.frames_in_listening = 0
                self.potential_silence_frames = 0
                self.was_recently_voiced = False
                self.estimated_noise_floor = self.vad_processing_settings.probability_threshold
                self.current_dynamic_vad_threshold = self.vad_processing_settings.probability_threshold

                if self.stt_recognizer: self.stt_recognizer.reset()
                else:
                    logger.error("STT recognizer not available after wake word detection!")
                    if not is_local_source and self.is_websocket_active: await self.send_error("Server STT error.")
                    self.state = "wakeword"
                    return

                if self.is_websocket_active:
                    await self.send_status("listening_started", "Listening...")
                else:
                     logger.info("Local: Listening started...")
                     if self.local_audio_output_engine and not self.local_audio_output_engine.is_running:
                         self.local_audio_output_engine.start()

        elif self.state == "listening":
            if not self.stt_recognizer:
                logger.error("STT recognizer not available in listening state!")
                if not is_local_source and self.is_websocket_active: await self.send_error("Server STT error.")
                self.state = "wakeword"
                return

            self.frames_in_listening += 1
            self.audio_buffer.extend(audio_chunk)

            self.stt_recognizer.accept_waveform(audio_chunk)
            partial_result_json = self.stt_recognizer.partial_result()
            partial_transcript = json.loads(partial_result_json).get("partial", "")
            if partial_transcript and self.is_websocket_active:
                await self.send_transcript(partial_transcript, is_final=False)

            voice_probability = self.vad_engine.process(pcm)
            active_threshold: float

            if self.vad_processing_settings.dynamic_threshold_enabled:
                if voice_probability < self.current_dynamic_vad_threshold * 0.7:
                    self.estimated_noise_floor = (1 - self.vad_processing_settings.noise_floor_alpha) * self.estimated_noise_floor + \
                                                 self.vad_processing_settings.noise_floor_alpha * voice_probability
                    self.estimated_noise_floor = max(0.0, min(self.estimated_noise_floor, 1.0))
                self.current_dynamic_vad_threshold = self.estimated_noise_floor + self.vad_processing_settings.threshold_margin_factor
                self.current_dynamic_vad_threshold = max(self.vad_processing_settings.min_dynamic_threshold,
                                                         min(self.current_dynamic_vad_threshold, self.vad_processing_settings.max_dynamic_threshold))
                active_threshold = self.current_dynamic_vad_threshold
            else:
                active_threshold = self.vad_processing_settings.probability_threshold

            is_voiced = voice_probability > active_threshold

            if is_voiced:
                self.silence_frames_count = 0
                self.potential_silence_frames = 0
                self.was_recently_voiced = True
            else:
                if self.was_recently_voiced:
                    self.potential_silence_frames += 1
                    if self.potential_silence_frames > self.vad_processing_settings.speech_hangover_frames:
                        self.silence_frames_count = self.potential_silence_frames - self.vad_processing_settings.speech_hangover_frames
                else:
                    self.silence_frames_count += 1

            grace_over = self.frames_in_listening >= self.vad_processing_settings.min_listening_frames
            silence_met = self.silence_frames_count >= self.vad_processing_settings.silence_frames_threshold
            max_len_met = self.frames_in_listening >= self.vad_processing_settings.max_listening_frames

            trigger_finalization = False
            if max_len_met:
                logger.info(f"Max listening frames ({self.frames_in_listening}) reached, finalizing.")
                trigger_finalization = True
            elif grace_over and silence_met:
                logger.info(f"VAD detected end of speech (silence frames: {self.silence_frames_count} >= {self.vad_processing_settings.silence_frames_threshold}).")
                trigger_finalization = True

            if trigger_finalization:
                final_result_json = self.stt_recognizer.final_result()
                final_transcript = json.loads(final_result_json).get("text", "").strip()
                logger.info(f"Final Transcript: '{final_transcript}'")

                if self.is_websocket_active:
                    await self.send_transcript(final_transcript, is_final=True)

                if final_transcript:
                    if self.llm_tts_task and not self.llm_tts_task.done():
                        logger.warning("New final transcript received while previous LLM/TTS task was running. Cancelling old task.")
                        self.llm_tts_task.cancel()
                    self.llm_tts_task = asyncio.create_task(
                        self._run_llm_tts_or_offline(final_transcript, thread_id)
                    )
                else: # No final transcript (e.g. only silence)
                    self.state = "wakeword"
                    if self.is_websocket_active:
                        await self.send_status("wakeword_listening", "No speech detected. Waiting for wake word...")
                    else:
                        logger.info("Local: No speech detected. Waiting for wake word...")

                self.audio_buffer.clear()
                self.silence_frames_count = 0
                self.frames_in_listening = 0
                self.potential_silence_frames = 0
                self.was_recently_voiced = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global manager
    if not manager:
        logger.error("ConnectionManager not initialized for WebSocket endpoint.")
        await websocket.accept()
        await websocket.close(code=1011, reason="Server not ready")
        return

    connected = await manager.connect(websocket)
    if not connected:
        logger.info("WebSocket connection attempt failed or was rejected by ConnectionManager.")
        return # connect() already handled closing the websocket if it was accepted then rejected

    try:
        while True: # manager.is_websocket_active and manager.websocket: # Check if manager still considers it active
            # The loop condition `while True` is fine as receive_json will raise on disconnect.
            message = await websocket.receive_json()
            if message.get("type") == "audio_chunk":
                audio_b64 = message.get("data")
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        # Ensure manager is still valid and websocket is the current one
                        if manager and manager.websocket is websocket and manager.is_websocket_active:
                             await manager.process_audio(audio_bytes, ASSISTANT_THREAD_ID, is_local_source=False)
                        else:
                            logger.warning("Received audio_chunk but manager state is inconsistent or websocket changed.")
                            break # Exit loop if manager state is no longer valid for this socket
                    except (base64.binascii.Error, TypeError) as e:
                        logger.error(f"Invalid base64 audio data: {e}")
                        if manager and manager.websocket is websocket and manager.is_websocket_active:
                             await manager.send_error("Invalid audio data.")
                    except Exception as e:
                         logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
                         if manager and manager.websocket is websocket and manager.is_websocket_active:
                              await manager.send_error("Server error processing audio.")
            # Add other message type handlers here if client sends more than audio
            # e.g., client settings, explicit stop commands, etc.
            
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected by client: code={e.code}, reason='{e.reason}'")
        # manager.disconnect will be called in finally
    except Exception as e:
        logger.error(f"Unexpected WebSocket Error: {e}", exc_info=True)
        if manager and manager.websocket is websocket and manager.is_websocket_active:
            # Try to send an error message before disconnecting
            try:
                await manager.send_error(f"Unexpected server-side WebSocket error: {str(e)}")
            except Exception as send_err:
                logger.error(f"Failed to send error to client before disconnect: {send_err}")
        # manager.disconnect will be called in finally
    finally:
        # Ensure disconnect is called, passing the websocket that triggered this endpoint
        if manager and manager.websocket is websocket: # Only disconnect if this is the active websocket
            logger.info(f"Calling manager.disconnect for websocket {websocket.client}")
            await manager.disconnect(reason="WebSocket endpoint cleanup")
        else:
            logger.info(f"Manager.websocket is no longer {websocket.client} or manager is None; disconnect might have been handled.")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Lumi Voice Assistant server (Modular) on {settings.webapp.host}:{settings.webapp.port}")
    
    if settings.sounddevice.enabled:
        try:
            import sounddevice as sd
            logger.info("Available Audio Devices (for SoundDevice engine):")
            logger.info(f"\n{sd.query_devices()}")
            logger.info(f"Using Input Device Index: {settings.sounddevice.input_device_index if settings.sounddevice.input_device_index is not None else 'Default'}")
            logger.info(f"Using Output Device Index: {settings.sounddevice.output_device_index if settings.sounddevice.output_device_index is not None else 'Default'}")
            logger.info(f"SoundDevice TTS Output Sample Rate (default): {settings.sounddevice.tts_output_sample_rate} Hz")
        except Exception as e:
            logger.warning(f"Could not list audio devices on startup: {e}")
    
    if settings.engines.tts_engine == "hybrid" or settings.engines.tts_engine == "gemini" or \
       settings.engines.tts_online_provider == "gemini" or settings.engines.tts_offline_provider == "gemini":
        logger.info(f"Gemini TTS configured with model: {settings.gemini_tts.model}, voice: {settings.gemini_tts.voice_name}, sample_rate: {settings.gemini_tts.sample_rate}Hz")
        if not settings.gemini_tts.api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini TTS will not function.")
        else:
            logger.info("GEMINI_API_KEY is set.")

    if settings.audio.play_activation_sound:
        if settings.audio.activation_sound_path: # Path is resolved and checked in settings
            logger.info(f"Activation sound enabled and file found: {settings.audio.activation_sound_path}")
        else:
            logger.warning("Activation sound enabled but ACTIVATION_SOUND_PATH is not set or file not found.")
    else:
        logger.info("Activation sound is disabled.")

    uvicorn.run("main:app", host=settings.webapp.host, port=settings.webapp.port, reload=True)