import asyncio
import base64
import json
import logging
import os
import struct
import wave # Added for loading activation sound
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Tuple # Added Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# Project specific imports
from settings import settings, Settings, AudioSettings, VADSettings # Import the main Settings class and AudioSettings for type hint
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

# Global engine instances
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

# Connection manager instance
manager: Optional["ConnectionManager"] = None # Forward declaration for ConnectionManager

# Assistant's conversation thread ID for LLM context
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
                    process_audio_callback=None, # Will be set later by initialize_global_engines
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
        from tools.device_control import initialize_device_control_tool # Import here to avoid circular dependency if tool uses settings
        initialize_device_control_tool(comm_service)
    else:
        logger.error("Communication service failed to initialize. Dependent services will be affected.")

    if comm_service: # Offline processor depends on communication service
        offline_command_processor = DefaultOfflineCommandProcessor(comm_service=comm_service)
        logger.info("DefaultOfflineCommandProcessor initialized.")
    else:
        offline_command_processor = None
        logger.warning("OfflineCommandProcessor not initialized as comm_service is unavailable.")

    # Initialize core processing engines
    wake_word_engine = create_engine_instance("wake_word", app_settings.engines.wake_word_engine, app_settings)
    vad_engine = create_engine_instance("vad", app_settings.engines.vad_engine, app_settings)
    stt_engine = create_engine_instance("stt", app_settings.engines.stt_engine, app_settings)
    tts_engine = create_engine_instance("tts", app_settings.engines.tts_engine, app_settings)
    nlu_engine = create_engine_instance("nlu", app_settings.engines.nlu_engine, app_settings)
    llm_logic_engine = create_engine_instance("llm_logic", app_settings.engines.llm_logic_engine, app_settings)
    offline_llm_logic_engine = create_engine_instance("offline_llm_logic", app_settings.engines.offline_llm_engine, app_settings)
    
    # Log status of critical engines
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
    
    # Initialize ConnectionManager (which uses the engines)
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

    # Initialize Audio Input Engine and set its callback
    _audio_input_engine_instance = create_engine_instance("audio_input", app_settings.engines.audio_input_engine, app_settings)
    if _audio_input_engine_instance and isinstance(_audio_input_engine_instance, AudioInputEngineBase):
        async def process_audio_wrapper(audio_chunk: bytes):
            if manager: # manager should be initialized by now
                await manager.process_audio(audio_chunk, is_local_source=True)
        _audio_input_engine_instance._process_audio_cb = process_audio_wrapper # type: ignore[attr-defined]
        audio_input_engine = _audio_input_engine_instance
        logger.info(f"AudioInputEngine ({app_settings.engines.audio_input_engine}) created.")
        if audio_input_engine and audio_input_engine.is_enabled:
            if wake_word_engine and audio_input_engine.sample_rate != wake_word_engine.sample_rate:
                 logger.warning(f"AudioInput SR ({audio_input_engine.sample_rate}) != WakeWord SR ({wake_word_engine.sample_rate}).")
            if wake_word_engine and audio_input_engine.frame_length != wake_word_engine.frame_length:
                 logger.warning(f"AudioInput frame len ({audio_input_engine.frame_length}) != WakeWord frame len ({wake_word_engine.frame_length}).")
        else:
            logger.info(f"AudioInputEngine ({app_settings.engines.audio_input_engine}) is disabled or failed to initialize.")
    else:
        logger.warning(f"Could not create or wrong type for AudioInputEngine: {app_settings.engines.audio_input_engine}")

    # Initialize Audio Output Engine
    audio_output_engine = create_engine_instance("audio_output", app_settings.engines.audio_output_engine, app_settings)
    if audio_output_engine and audio_output_engine.is_enabled:
        logger.info(f"AudioOutputEngine ({app_settings.engines.audio_output_engine}) created and enabled.")
        if manager: manager.set_local_audio_output(audio_output_engine)
    else:
        logger.info(f"AudioOutputEngine ({app_settings.engines.audio_output_engine}) is disabled or failed to initialize.")

    # Startup for engines that require it
    if tts_engine: await tts_engine.startup()
    if llm_logic_engine: await llm_logic_engine.startup()
    if offline_llm_logic_engine: await offline_llm_logic_engine.startup()

    logger.info("Global engines initialization phase complete.")


async def shutdown_global_engines():
    logger.info("Shutting down global engines...")
    # Shutdown in reverse order of dependency or specific order if needed
    engine_list = [
        audio_input_engine, audio_output_engine, # IO first
        tts_engine, llm_logic_engine, offline_llm_logic_engine, nlu_engine,
        stt_engine, vad_engine, wake_word_engine, 
        comm_service # Communication last or as appropriate
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
    await start_connectivity_monitoring()
    try:
        init_scheduler_db()
        init_music_likes_table()
    except Exception as e:
        logger.error(f"Failed to initialize scheduler or music database: {e}. Some features might not work.", exc_info=True)

    await initialize_global_engines(settings) # settings is the global Settings instance

    # Start reminder checker if dependencies are met
    if tts_engine and audio_output_engine and await tts_engine.is_healthy() and audio_output_engine.is_enabled:
        if not settings.scheduler_db_path:
            logger.warning("Scheduler DB path not set. Reminder checker might not function correctly.")
        await start_reminder_checker(tts_engine, audio_output_engine)
    else:
        logger.warning("TTS or Audio Output engine not available/healthy, reminder checker not started.")

    # Start local audio I/O if no WebSocket is active initially
    if audio_input_engine and audio_input_engine.is_enabled and \
       audio_output_engine and audio_output_engine.is_enabled and \
       manager and not manager.is_websocket_active: # manager should be initialized
        logger.info("Lifespan: No active WebSocket, starting local audio I/O.")
        if not audio_output_engine.is_running: audio_output_engine.start()
        if not audio_input_engine.is_running: audio_input_engine.start()
        manager.state = "wakeword"
        logger.info("Lifespan: Local audio interface activated, waiting for wake word.")
    
    logger.info("Lifespan: Startup phase complete.")
    yield
    logger.info("Lifespan: Shutdown phase beginning...")
    await stop_reminder_checker()
    await shutdown_global_engines()
    await stop_connectivity_monitoring()
    logger.info("Lifespan: Shutdown phase complete.")


app = FastAPI(
    title="Lumi Voice Assistant Backend (Modular)",
    lifespan=lifespan
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
    logger.info(f"Created static directory at: {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_root():
    static_file_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(static_file_path):
        logger.error(f"Frontend file not found at: {static_file_path}")
        raise HTTPException(status_code=404, detail=f"Frontend file 'index.html' not found in '{static_dir}'.")
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
                 vad_processing_settings: VADSettings):

        self.wake_word_engine = wake_word_engine
        self.vad_engine = vad_engine
        self.stt_provider = stt_provider
        self.tts_engine = tts_engine
        self.nlu_engine = nlu_engine
        self.llm_logic_engine = llm_logic_engine
        self.offline_llm_logic_engine = offline_llm_logic_engine
        self.comm_service = comm_service
        self.offline_processor = offline_processor
        self.global_audio_settings = global_audio_settings
        self.vad_processing_settings = vad_processing_settings

        self._last_mentioned_device_for_pronoun: Optional[str] = None # For pronoun resolution in offline commands

        self.websocket: Optional[WebSocket] = None
        self.stt_recognizer: Optional[STTEngineBase.RecognizerInstance] = None
        self.state: str = "disconnected" # States: disconnected, wakeword, listening, processing, speaking
        self.audio_buffer: bytearray = bytearray()
        self.silence_frames_count: int = 0
        self.frames_in_listening: int = 0
        self.intelligent_response_task: Optional[asyncio.Task] = None # Task for LLM/NLU + TTS
        self.is_websocket_active: bool = False
        self.local_audio_output_engine: Optional[AudioOutputEngineBase] = None

        # VAD dynamic threshold parameters
        self.estimated_noise_floor: float = self.vad_processing_settings.probability_threshold
        self.current_dynamic_vad_threshold: float = self.vad_processing_settings.probability_threshold
        self.potential_silence_frames: int = 0 # For hangover logic
        self.was_recently_voiced: bool = False # For hangover logic

        # Activation sound resources
        self.activation_sound_bytes: Optional[bytes] = None
        self.activation_sound_sample_rate: Optional[int] = None
        self._load_activation_sound()

        # Sanity checks for engine frame lengths
        if self.wake_word_engine and self.global_audio_settings.frame_length != self.wake_word_engine.frame_length:
            logger.warning(f"Global audio frame length ({self.global_audio_settings.frame_length}) differs from WakeWord frame length ({self.wake_word_engine.frame_length}).")
        if self.vad_engine and self.global_audio_settings.frame_length != self.vad_engine.frame_length:
            logger.warning(f"Global audio frame length ({self.global_audio_settings.frame_length}) differs from VAD frame length ({self.vad_engine.frame_length}).")

        # Initialize STT recognizer instance
        if self.stt_provider:
            try:
                self.stt_recognizer = self.stt_provider.create_recognizer()
                logger.info("ConnectionManager: STT recognizer instance created during init.")
            except Exception as e:
                logger.error(f"ConnectionManager: Failed to create STT recognizer during init: {e}", exc_info=True)
        else:
            logger.warning("ConnectionManager: STT provider not available during init. STT will not work.")

    def _load_activation_sound(self):
        if not self.global_audio_settings.play_activation_sound: return
        sound_path = self.global_audio_settings.activation_sound_path
        if not sound_path:
            logger.warning("Activation sound is enabled but no path is specified (ACTIVATION_SOUND_PATH).")
            return
        if not os.path.exists(sound_path):
            logger.warning(f"Activation sound file not found: {sound_path}. Activation sound will not be played.")
            return
        try:
            with wave.open(sound_path, 'rb') as wf:
                self.activation_sound_bytes = wf.readframes(wf.getnframes())
                self.activation_sound_sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                logger.info(f"Activation sound loaded: {sound_path}, SR={self.activation_sound_sample_rate}Hz, Channels={channels}, Width={sampwidth}bytes, Size={len(self.activation_sound_bytes)} bytes")
                if channels != 1:
                    logger.warning(f"Activation sound '{sound_path}' is not mono. Playback might be suboptimal.")
                if sampwidth != 2: # Assuming 16-bit PCM
                    logger.warning(f"Activation sound '{sound_path}' sample width is {sampwidth} bytes (expected 2 for int16). Playback might be incorrect.")
        except Exception as e:
            logger.error(f"Failed to load activation sound from {sound_path}: {e}", exc_info=True)
            self.activation_sound_bytes = None # Ensure it's None on failure

    def set_local_audio_output(self, local_audio_out_engine: AudioOutputEngineBase):
        self.local_audio_output_engine = local_audio_out_engine

    async def connect(self, websocket: WebSocket):
        global audio_input_engine, audio_output_engine # To control local audio I/O
        if self.websocket is not None and self.websocket.client_state == WebSocketState.CONNECTED:
            logger.warning("Another WebSocket client tried to connect while one is active. Rejecting.")
            await websocket.accept()
            await websocket.close(code=1008, reason="Server busy. Another client is connected.")
            return False

        await websocket.accept()
        self.websocket = websocket
        self.state = "wakeword"
        self._reset_listening_state()
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
            await self.disconnect(code=1011, reason="Server STT error")
            return False
        
        try:
            if not self.stt_recognizer:
                logger.warning("STT recognizer was not created in __init__, attempting now.")
                self.stt_recognizer = self.stt_provider.create_recognizer()
            self.stt_recognizer.reset() # Always reset on new connection
            logger.info("STT recognizer (re)initialized and reset on connect.")
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

        if self.intelligent_response_task and not self.intelligent_response_task.done():
            self.intelligent_response_task.cancel()
            try: await self.intelligent_response_task
            except asyncio.CancelledError: logger.info("Intelligent response task cancelled on disconnect.")
            except Exception as e: logger.error(f"Error during intelligent response task cancellation on disconnect: {e}")
        self.intelligent_response_task = None

        current_ws = self.websocket
        self.websocket = None # Nullify before trying to close
        if current_ws and current_ws.client_state == WebSocketState.CONNECTED:
            try:
                logger.info(f"Closing WebSocket with code {code} and reason '{reason}'")
                await current_ws.close(code=code, reason=reason)
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        
        old_state = self.state
        self.state = "disconnected"
        logger.info(f"WebSocket client disconnected (previous state: {old_state}). Code: {code}, Reason: {reason}")

        # Resume local audio if enabled
        if audio_input_engine and audio_input_engine.is_enabled:
            logger.info("WebSocket disconnected, resuming local audio input.")
            if not audio_input_engine.is_running: audio_input_engine.start()
            audio_input_engine.resume()
            self.state = "wakeword" # Transition back to wakeword for local mode
            logger.info("Local audio interface active, waiting for wake word.")
        if audio_output_engine and audio_output_engine.is_enabled:
            if not audio_output_engine.is_running: audio_output_engine.start()

    async def _send_json(self, data: dict) -> bool:
        if self.is_websocket_active and self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await self.websocket.send_json(data)
                return True
            except WebSocketDisconnect:
                logger.warning("WebSocket disconnected while trying to send JSON.")
                # Schedule disconnect, don't await it here to avoid deadlocks if called from disconnect path
                asyncio.create_task(self.disconnect(reason="WebSocket disconnected during send"))
                return False
            except Exception as e:
                 logger.error(f"Error sending JSON to client: {e}")
                 return False
        return False

    async def send_status(self, status_code: str, message: str):
        await self._send_json({"type": "status", "code": status_code, "message": message})

    async def send_transcript(self, transcript: str, is_final: bool):
         await self._send_json({"type": "transcript", "text": transcript, "is_final": is_final})

    async def send_tts_info(self, sample_rate: int, channels: int = 1, bit_depth: int = 16):
        await self._send_json({"type": "tts_info", "sample_rate": sample_rate, "channels": channels, "bit_depth": bit_depth})

    async def send_tts_chunk(self, audio_chunk_b64: str):
        await self._send_json({"type": "tts_chunk", "data": audio_chunk_b64})

    async def send_tts_finished(self):
        await self._send_json({"type": "tts_finished"})

    async def send_error(self, error_message: str):
        logger.error(f"Sending error to client: {error_message}")
        await self._send_json({"type": "error", "message": error_message})

    def _reset_listening_state(self):
        self.audio_buffer.clear()
        self.silence_frames_count = 0
        self.frames_in_listening = 0
        self.potential_silence_frames = 0
        self.was_recently_voiced = False
        # Reset VAD noise floor estimation to default or a bit higher than min
        self.estimated_noise_floor = self.vad_processing_settings.min_dynamic_threshold * 1.2 \
            if self.vad_processing_settings.dynamic_threshold_enabled \
            else self.vad_processing_settings.probability_threshold
        self.current_dynamic_vad_threshold = self.estimated_noise_floor + self.vad_processing_settings.threshold_margin_factor

    async def _get_active_tts_synthesizer(self) -> Optional[TTSEngineBase]:
        if not self.tts_engine: return None
        if isinstance(self.tts_engine, HybridTTSEngine):
            return await self.tts_engine.get_active_engine_for_synthesis()
        # Check if it's a standalone synthesizable engine (has get_output_sample_rate)
        elif hasattr(self.tts_engine, 'get_output_sample_rate') and hasattr(self.tts_engine, 'synthesize_stream'):
            return self.tts_engine
        logger.warning(f"TTS engine {self.tts_engine.__class__.__name__} does not appear to be a valid synthesizer.")
        return None

    async def _get_intelligent_response(self, text: str, thread_id: str) -> Tuple[str, Optional[str]]:
        """
        Determines the response text using online/offline LLMs or NLU.
        Returns: (response_text, updated_pronoun_context)
        """
        response_text = ""
        new_pronoun_context: Optional[str] = None
        online_capable = await is_internet_available()
        attempt_online_llm = online_capable and settings.ai.online_mode and self.llm_logic_engine
        online_llm_succeeded = False
        online_llm_attempt_made = False

        if attempt_online_llm:
            online_llm_attempt_made = True
            logger.info("Attempting online LLM.")
            try:
                lumi_response = await self.llm_logic_engine.ask(text, thread_id=thread_id) # type: ignore
                candidate_response = lumi_response if isinstance(lumi_response, str) else lumi_response.content # type: ignore
                
                if candidate_response and not (
                    candidate_response.startswith("Sorry, an error occurred") or \
                    candidate_response.startswith("Sorry, the language model is not available")
                    ):
                    response_text = candidate_response
                    online_llm_succeeded = True
                    logger.info("Online LLM successfully provided a response.")
                else:
                    logger.warning(f"Online LLM returned an empty or error-like response: '{candidate_response}'. Falling back.")
            except Exception as e:
                logger.error(f"Exception during online LLM call: {e}. Falling back.", exc_info=True)
        
        if not online_llm_succeeded:
            if online_llm_attempt_made: logger.info("Online LLM failed or not used. Proceeding with offline processing.")
            else: logger.info(f"Offline processing path. Internet: {online_capable}, Online Mode: {settings.ai.online_mode}.")

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
                    if resolved_command.get("executable") and resolved_command.get("resolved_device_name_for_context_update"):
                        new_pronoun_context = resolved_command["resolved_device_name_for_context_update"]
                else:
                    logger.warning("Offline NLU parsed a command, but no offline_processor is available.")
                    response_text = "I understood an offline command, but cannot process it right now."
            elif self.offline_llm_logic_engine: # NLU did not match or unavailable, try offline LLM
                logger.info("NLU did not identify a command or NLU unavailable. Trying offline LLM.")
                try:
                    lumi_response_offline = await self.offline_llm_logic_engine.ask(text, thread_id=thread_id)
                    candidate_offline_response = lumi_response_offline if isinstance(lumi_response_offline, str) else lumi_response_offline.content # type: ignore
                    if candidate_offline_response and not (
                        candidate_offline_response.startswith("Sorry, an error occurred") or \
                        candidate_offline_response.startswith("Sorry, the offline language model is not available")
                        ):
                        response_text = candidate_offline_response
                    else:
                        logger.warning(f"Offline LLM also returned an empty or error-like response: '{candidate_offline_response}'")
                except Exception as e:
                    logger.error(f"Exception during offline LLM call: {e}.", exc_info=True)
            
            if not response_text: # If still no response after offline attempts
                fallback_message = "I couldn't process this request with available offline tools."
                if online_llm_attempt_made: # Online was tried and failed
                    fallback_message = "I couldn't process this online, and no suitable offline action was found."
                elif not online_capable and settings.ai.online_mode:
                     fallback_message = "I'm offline. I can only process specific commands, and this wasn't one of them."
                response_text = fallback_message
        
        if not response_text: # Final safety net
            logger.error("All processing paths failed to produce a response_text.")
            response_text = "I'm sorry, I was unable to process your request at this time."
        
        return response_text, new_pronoun_context

    async def _synthesize_and_send_tts_response(self, response_text: str):
        if not response_text:
            logger.warning("TTS synthesis skipped: No response text provided.")
            return

        active_tts_synthesizer = await self._get_active_tts_synthesizer()
        if not active_tts_synthesizer or not await active_tts_synthesizer.is_healthy():
            logger.error("TTS: No active and healthy synthesizer found or TTS engine is not correctly configured.")
            if self.is_websocket_active: await self.send_error("Sorry, I can't speak right now (TTS synth error).")
            return

        current_tts_sample_rate = active_tts_synthesizer.get_output_sample_rate()
        logger.info(f"TTS: Using {active_tts_synthesizer.__class__.__name__} with SR {current_tts_sample_rate}Hz.")

        if self.is_websocket_active:
            await self.send_tts_info(sample_rate=current_tts_sample_rate)
            await self.send_status("speaking_started", "Speaking...")
        else:
            logger.info(f"Synthesizing for local playback (SR: {current_tts_sample_rate}Hz): '{response_text[:50]}...'")
        
        self.state = "speaking"
        tts_audio_bytes_for_local = bytearray()

        try:
            async for tts_chunk in active_tts_synthesizer.synthesize_stream(response_text):
                if self.is_websocket_active:
                    await self.send_tts_chunk(base64.b64encode(tts_chunk).decode('utf-8'))
                if self.local_audio_output_engine and self.local_audio_output_engine.is_enabled:
                    tts_audio_bytes_for_local.extend(tts_chunk)
            
            if self.is_websocket_active: await self.send_tts_finished()
            
            if tts_audio_bytes_for_local and self.local_audio_output_engine and \
               self.local_audio_output_engine.is_enabled and not self.is_websocket_active:
                logger.info(f"Queueing {len(tts_audio_bytes_for_local)} bytes for local TTS (SR: {current_tts_sample_rate}Hz).")
                self.local_audio_output_engine.play_tts_bytes(
                    bytes(tts_audio_bytes_for_local),
                    sample_rate=current_tts_sample_rate
                )
        except Exception as e:
            logger.error(f"Error during TTS synthesis or playback: {e}", exc_info=True)
            if self.is_websocket_active: await self.send_error("Sorry, an error occurred while speaking.")


    async def _process_intelligent_response_flow(self, text: str, thread_id: str):
        """Orchestrates getting a response and then synthesizing/sending it."""
        try:
            if self.is_websocket_active:
                await self.send_status("processing_started", "Thinking...")
            else:
                logger.info(f"Processing request (local audio) for text: '{text[:50]}...'")
            self.state = "processing"

            response_text, new_pronoun_context = await self._get_intelligent_response(text, thread_id)
            
            if new_pronoun_context:
                self._last_mentioned_device_for_pronoun = new_pronoun_context
                logger.debug(f"ConnectionManager: Updated pronoun context to '{self._last_mentioned_device_for_pronoun}'")

            await self._synthesize_and_send_tts_response(response_text)

        except asyncio.CancelledError:
            logger.info("Intelligent response processing task cancelled.")
        except Exception as e:
            logger.error(f"Error in _process_intelligent_response_flow: {e}", exc_info=True)
            if self.is_websocket_active: 
                await self.send_error(f"Error processing request: {str(e)}")
        finally:
            if self.state != "disconnected": 
                self.state = "wakeword"
                if self.is_websocket_active:
                    await self.send_status("wakeword_listening", "Waiting for wake word...")
                else:
                    logger.info("Local: Waiting for wake word...")
            self.intelligent_response_task = None


    async def process_audio(self, audio_chunk: bytes, is_local_source: bool = False):
        if self.state in ["processing", "speaking", "disconnected"]: return

        if not all([self.wake_word_engine, self.vad_engine, self.stt_provider, self.stt_recognizer]):
            logger.error("Core audio processing engines not ready in ConnectionManager.")
            if not is_local_source and self.is_websocket_active: await self.send_error("Server audio engines not ready.")
            return

        bytes_per_sample = 2 # Assuming 16-bit audio
        expected_bytes = self.global_audio_settings.frame_length * bytes_per_sample

        if len(audio_chunk) != expected_bytes:
            logger.debug(f"Audio chunk size mismatch: got {len(audio_chunk)}, expected {expected_bytes}. Skipping.")
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
                self._handle_wake_word_detection(is_local_source)
                
        elif self.state == "listening":
            self._handle_listening_audio(pcm, audio_chunk, is_local_source)

    def _handle_wake_word_detection(self, is_local_source: bool):
        if self.global_audio_settings.play_activation_sound:
            if self.is_websocket_active:
                logger.info("Sending cue to client to play activation sound.")
                # Fire and forget, don't await here to block processing
                asyncio.create_task(self.send_status("play_activation_sound_cue", "Client should play activation sound."))
            elif self.activation_sound_bytes and self.activation_sound_sample_rate and \
                    self.local_audio_output_engine and self.local_audio_output_engine.is_enabled:
                logger.info(f"Playing activation sound locally (SR: {self.activation_sound_sample_rate}Hz).")
                self.local_audio_output_engine.play_tts_bytes(
                    self.activation_sound_bytes,
                    sample_rate=self.activation_sound_sample_rate
                )
        
        self.state = "listening"
        self._reset_listening_state()

        if not self.stt_recognizer: # Should have been initialized
            logger.error("STT recognizer not available after wake word detection!")
            if not is_local_source and self.is_websocket_active: 
                asyncio.create_task(self.send_error("Server STT error."))
            self.state = "wakeword" # Revert state
            return
        self.stt_recognizer.reset()

        status_message = "Listening..."
        if self.is_websocket_active:
            asyncio.create_task(self.send_status("listening_started", status_message))
        else:
            logger.info(f"Local: {status_message}")
            if self.local_audio_output_engine and not self.local_audio_output_engine.is_running:
                self.local_audio_output_engine.start() # Ensure output is ready if needed later

    def _handle_listening_audio(self, pcm_frame: tuple, audio_chunk: bytes, is_local_source: bool):
        if not self.stt_recognizer: # Should be available
            logger.error("STT recognizer not available in listening state!")
            if not is_local_source and self.is_websocket_active: asyncio.create_task(self.send_error("Server STT error."))
            self.state = "wakeword"
            return

        self.frames_in_listening += 1
        # self.audio_buffer.extend(audio_chunk) # Buffering for resending if needed, consider if STT needs full audio at end

        self.stt_recognizer.accept_waveform(audio_chunk)
        partial_result_json = self.stt_recognizer.partial_result()
        partial_transcript = json.loads(partial_result_json).get("partial", "")
        if partial_transcript and self.is_websocket_active:
            asyncio.create_task(self.send_transcript(partial_transcript, is_final=False))

        # VAD processing
        voice_probability = self.vad_engine.process(pcm_frame) # type: ignore
        active_threshold: float

        if self.vad_processing_settings.dynamic_threshold_enabled:
            # Update noise floor estimate if current frame is likely non-speech
            if voice_probability < self.current_dynamic_vad_threshold * 0.7: # Heuristic: significantly below current threshold
                self.estimated_noise_floor = (1 - self.vad_processing_settings.noise_floor_alpha) * self.estimated_noise_floor + \
                                                self.vad_processing_settings.noise_floor_alpha * voice_probability
                self.estimated_noise_floor = max(0.0, min(self.estimated_noise_floor, 1.0)) # Clamp
            
            # Calculate current dynamic threshold based on estimated noise floor
            self.current_dynamic_vad_threshold = self.estimated_noise_floor + self.vad_processing_settings.threshold_margin_factor
            self.current_dynamic_vad_threshold = max(self.vad_processing_settings.min_dynamic_threshold,
                                                        min(self.current_dynamic_vad_threshold, self.vad_processing_settings.max_dynamic_threshold))
            active_threshold = self.current_dynamic_vad_threshold
        else:
            active_threshold = self.vad_processing_settings.probability_threshold

        is_voiced = voice_probability > active_threshold

        if is_voiced:
            self.silence_frames_count = 0
            self.potential_silence_frames = 0 # Reset potential silence counter
            self.was_recently_voiced = True
        else: # Not voiced
            if self.was_recently_voiced: # If speech just ended, start counting hangover frames
                self.potential_silence_frames += 1
                if self.potential_silence_frames > self.vad_processing_settings.speech_hangover_frames:
                    # After hangover, actual silence frames start accumulating
                    self.silence_frames_count = self.potential_silence_frames - self.vad_processing_settings.speech_hangover_frames
            else: # Was not recently voiced, just count silence
                self.silence_frames_count += 1
        
        # Determine if speech has ended
        min_listen_frames_passed = self.frames_in_listening >= self.vad_processing_settings.min_listening_frames
        silence_threshold_met = self.silence_frames_count >= self.vad_processing_settings.silence_frames_threshold
        max_listen_frames_met = self.frames_in_listening >= self.vad_processing_settings.max_listening_frames

        trigger_finalization = False
        if max_listen_frames_met:
            logger.info(f"Max listening frames ({self.frames_in_listening}) reached, finalizing.")
            trigger_finalization = True
        elif min_listen_frames_passed and silence_threshold_met:
            logger.info(f"VAD detected end of speech (silence frames: {self.silence_frames_count} >= {self.vad_processing_settings.silence_frames_threshold} after hangover).")
            trigger_finalization = True

        if trigger_finalization:
            final_result_json = self.stt_recognizer.final_result()
            final_transcript = json.loads(final_result_json).get("text", "").strip()
            logger.info(f"Final Transcript: '{final_transcript}'")

            if self.is_websocket_active:
                asyncio.create_task(self.send_transcript(final_transcript, is_final=True))

            if final_transcript:
                if self.intelligent_response_task and not self.intelligent_response_task.done():
                    logger.warning("New final transcript received while previous intelligent response task was running. Cancelling old task.")
                    self.intelligent_response_task.cancel()
                # Use the global ASSISTANT_THREAD_ID for conversation context
                self.intelligent_response_task = asyncio.create_task(
                    self._process_intelligent_response_flow(final_transcript, ASSISTANT_THREAD_ID)
                )
            else: # No final transcript (e.g. only silence or VAD timeout before speech)
                self.state = "wakeword"
                message = "No speech detected. Waiting for wake word..."
                if self.is_websocket_active:
                    asyncio.create_task(self.send_status("wakeword_listening", message))
                else:
                    logger.info(f"Local: {message}")
            
            self._reset_listening_state() # Also clears audio_buffer


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global manager # manager should be initialized by lifespan
    if not manager:
        logger.error("ConnectionManager not initialized for WebSocket endpoint.")
        await websocket.accept()
        await websocket.close(code=1011, reason="Server not ready")
        return

    connected = await manager.connect(websocket)
    if not connected:
        logger.info("WebSocket connection attempt failed or was rejected by ConnectionManager.")
        return 

    try:
        while manager.is_websocket_active and manager.websocket is websocket: # Ensure we're acting on the current WebSocket
            message = await websocket.receive_json() # This will raise WebSocketDisconnect if client closes
            
            if message.get("type") == "audio_chunk":
                audio_b64 = message.get("data")
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        # Double check manager and websocket validity before processing
                        if manager.is_websocket_active and manager.websocket is websocket:
                             await manager.process_audio(audio_bytes, is_local_source=False)
                        else:
                            logger.warning("Received audio_chunk but manager state or websocket changed. Breaking loop.")
                            break 
                    except (base64.binascii.Error, TypeError) as e:
                        logger.error(f"Invalid base64 audio data: {e}")
                        if manager.is_websocket_active and manager.websocket is websocket:
                             await manager.send_error("Invalid audio data.")
                    except Exception as e:
                         logger.error(f"Error processing WebSocket audio_chunk message: {e}", exc_info=True)
                         if manager.is_websocket_active and manager.websocket is websocket:
                              await manager.send_error("Server error processing audio.")
            # Add other message type handlers here if needed (e.g., client settings, explicit stop commands)
            
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected by client: code={e.code}, reason='{e.reason}'")
        # manager.disconnect will be called in finally if this was the active websocket
    except Exception as e:
        logger.error(f"Unexpected WebSocket Error: {e}", exc_info=True)
        if manager and manager.is_websocket_active and manager.websocket is websocket:
            try:
                await manager.send_error(f"Unexpected server-side WebSocket error: {str(e)}")
            except Exception as send_err:
                logger.error(f"Failed to send error to client before disconnect: {send_err}")
    finally:
        # Ensure disconnect is called only if this endpoint's websocket was the active one
        if manager and manager.websocket is websocket: 
            logger.info(f"Calling manager.disconnect for websocket {websocket.client} from endpoint finally block.")
            await manager.disconnect(reason="WebSocket endpoint cleanup")
        else:
            # This can happen if a new connection replaced the old one, and the old one's finally block runs later.
            logger.info(f"Manager.websocket is no longer {websocket.client} or manager is None; disconnect likely handled by new connection or previous call.")


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
        except Exception as e:
            logger.warning(f"Could not list audio devices on startup (sounddevice might be unavailable or misconfigured): {e}")
    
    if settings.engines.tts_engine == "hybrid" or settings.engines.tts_engine == "gemini" or \
       settings.engines.tts_online_provider == "gemini" or settings.engines.tts_offline_provider == "gemini":
        logger.info(f"Gemini TTS configured with model: {settings.gemini_tts.model}, voice: {settings.gemini_tts.voice_name}, sample_rate: {settings.gemini_tts.sample_rate}Hz")
        if not settings.gemini_tts.api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini TTS will not function.")

    if settings.audio.play_activation_sound and not settings.audio.activation_sound_path:
        logger.warning("Activation sound is enabled (PLAY_ACTIVATION_SOUND=True) but ACTIVATION_SOUND_PATH is not set. No sound will be played.")
    elif settings.audio.play_activation_sound:
         logger.info(f"Activation sound enabled. Will attempt to use: {settings.audio.activation_sound_path}")
    else:
        logger.info("Activation sound is disabled.")

    uvicorn.run("main:app", host=settings.webapp.host, port=settings.webapp.port, reload=True)