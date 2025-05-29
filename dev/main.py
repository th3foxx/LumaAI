import asyncio
import base64
import json
import logging
import os
import struct
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# Project specific imports
from settings import settings, Settings # Import the main Settings class
from connectivity import is_internet_available

from tools.scheduler import init_db as init_scheduler_db
from utils.music_db import init_music_likes_table
from services.reminder_checker import start_reminder_checker, stop_reminder_checker
# from offline_controller import parse_offline_command, execute_offline_command # Now part of NLU engine

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

# Engine factory (simple example)
# In a larger app, this could be more sophisticated, e.g., using plugin patterns
from engines.wake_word.picovoice_porcupine import PicovoicePorcupineEngine
from engines.vad.picovoice_cobra import PicovoiceCobraEngine
from engines.stt.vosk_stt import VoskSTTEngine
from engines.tts.paroli_tts import ParoliTTSEngine
from engines.tts.gemini_tts import GeminiTTSEngine # NEW
from engines.tts.hybrid_tts import HybridTTSEngine # NEW
from engines.nlu.rasa_nlu import RasaNLUEngine
from engines.llm_logic.langgraph_llm import LangGraphLLMEngine
from engines.llm_logic.ollama_llm import OllamaLLMEngine 
from engines.audio_io.sounddevice_io import SoundDeviceInputEngine, SoundDeviceOutputEngine
from engines.communication.mqtt_service import MQTTService
from engines.offline_processing.base import OfflineCommandProcessorBase
from engines.offline_processing.default_processor import DefaultOfflineCommandProcessor

# Tools for LLM (used by LangGraphLLMEngine, which imports them itself)
# from tools.device_control import set_device_attribute (tool itself uses comm_service)
# from tools.time import get_current_time

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Engine Instances ---
# These will be initialized in the lifespan manager
wake_word_engine: Optional[WakeWordEngineBase] = None
vad_engine: Optional[VADEngineBase] = None
stt_engine: Optional[STTRecognizerProvider] = None # Provides recognizer instances
tts_engine: Optional[TTSEngineBase] = None # This will be the HybridTTSEngine
nlu_engine: Optional[NLUEngineBase] = None
llm_logic_engine: Optional[LLMLogicEngineBase] = None
offline_llm_logic_engine: Optional[LLMLogicEngineBase] = None # For offline logic (e.g., Ollama)
audio_input_engine: Optional[AudioInputEngineBase] = None
audio_output_engine: Optional[AudioOutputEngineBase] = None
comm_service: Optional[CommunicationServiceBase] = None # e.g., MQTT
offline_command_processor: Optional[OfflineCommandProcessorBase] = None

# --- Connection Manager (WebSocket Handler) ---
# Forward declaration for type hint in process_audio_wrapper
class ConnectionManager: pass
manager: Optional[ConnectionManager] = None # Global instance, initialized in lifespan

ASSISTANT_THREAD_ID = "lumi-voice-assistant-session" # For LLM context

def create_engine_instance(engine_type: str, engine_name: str, global_settings: Settings) -> Optional[Any]: # Return type Any for flexibility
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
                    sample_rate=global_settings.vosk.sample_rate # Vosk specific sample rate
                )
        elif engine_type == "tts":
            if engine_name == "paroli":
                return ParoliTTSEngine(config=global_settings.paroli_server)
            elif engine_name == "gemini":
                # GeminiTTSEngine expects GeminiTTSSettings instance
                if not global_settings.gemini_tts.api_key: # Check API key early
                     logger.warning("Gemini TTS engine selected, but API key is missing in settings. It will likely fail.")
                return GeminiTTSEngine(config=global_settings.gemini_tts)
            elif engine_name == "hybrid":
                online_provider_name = global_settings.engines.tts_online_provider
                offline_provider_name = global_settings.engines.tts_offline_provider

                logger.info(f"Creating Hybrid TTS: Online provider='{online_provider_name}', Offline provider='{offline_provider_name}'")

                # Recursively create provider instances
                online_tts_instance = create_engine_instance("tts", online_provider_name, global_settings)
                offline_tts_instance = create_engine_instance("tts", offline_provider_name, global_settings)

                if not online_tts_instance:
                    logger.warning(f"Hybrid TTS: Online provider '{online_provider_name}' failed to initialize.")
                if not offline_tts_instance:
                    logger.warning(f"Hybrid TTS: Offline provider '{offline_provider_name}' failed to initialize.")
                
                if not online_tts_instance and not offline_tts_instance:
                    logger.error("Hybrid TTS: Both online and offline providers failed to initialize. Hybrid TTS will not be functional.")
                    return None # Critical failure for hybrid if no providers

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
                    process_audio_callback=None, 
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
    
    # TTS Engine (potentially Hybrid)
    tts_engine = create_engine_instance("tts", app_settings.engines.tts_engine, app_settings)
    
    nlu_engine = create_engine_instance("nlu", app_settings.engines.nlu_engine, app_settings)
    llm_logic_engine = create_engine_instance("llm_logic", app_settings.engines.llm_logic_engine, app_settings)
    offline_llm_logic_engine = create_engine_instance("offline_llm_logic", app_settings.engines.offline_llm_engine, app_settings)
    
    if not all([wake_word_engine, vad_engine, stt_engine, tts_engine, nlu_engine, comm_service]): 
        logger.critical("One or more core processing or communication engines failed to initialize. Functionality will be severely limited.")
    if not tts_engine: # Specifically check tts_engine after its creation attempt
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
        tts_engine=tts_engine, # This is now the Hybrid Engine
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
        _audio_input_engine_instance._process_audio_cb = process_audio_wrapper 
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

    # Startup for engines that need it (TTS server, LLM DB connections, MQTT)
    # comm_service already started up
    if tts_engine: await tts_engine.startup() # This will call startup on Hybrid, then on Gemini and Paroli
    if llm_logic_engine: await llm_logic_engine.startup()
    if offline_llm_logic_engine: await offline_llm_logic_engine.startup() # Ensure offline LLM also starts up if configured

    logger.info("Global engines initialization phase complete.")


async def shutdown_global_engines():
    logger.info("Shutting down global engines...")
    # Order might matter for dependencies, shutdown in reverse order of startup is a good heuristic
    engine_list = [
        # Higher level / composite engines first
        tts_engine, # Hybrid engine, will call shutdown on its components
        # LLMs
        llm_logic_engine, offline_llm_logic_engine,
        # NLU
        nlu_engine,
        # Audio processing pipeline components
        wake_word_engine, vad_engine, stt_engine, 
        # IO Engines
        audio_input_engine, audio_output_engine,
        # Communication
        comm_service 
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

    try:
        init_scheduler_db() 
        init_music_likes_table() 
    except Exception as e:
        logger.error(f"Failed to initialize scheduler or music database: {e}. Some features might not work.", exc_info=True)

    await initialize_global_engines(settings) 

    global tts_engine, audio_output_engine 
    if tts_engine and audio_output_engine and await tts_engine.is_healthy() and audio_output_engine.is_enabled: # Check tts_engine health
        if not settings.scheduler_db_path:
            logger.warning("Scheduler DB path not set. Reminder checker might not function correctly.")
        await start_reminder_checker(tts_engine, audio_output_engine) # tts_engine is Hybrid
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
    logger.info("Lifespan: Shutdown phase complete.")


app = FastAPI(
    title="Lumi Voice Assistant Backend (Modular)",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_root():
    static_file_path = os.path.join("static", "index.html")
    if not os.path.exists(static_file_path):
        raise HTTPException(status_code=404, detail="Frontend file not found.")
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
                 global_audio_settings: "AudioSettings", # type: ignore
                 vad_processing_settings: "VADSettings"): # type: ignore

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
        self.vad_processing_settings = vad_processing_settings # Уже есть

        self.websocket: Optional[WebSocket] = None
        self.stt_recognizer: Optional[STTEngineBase.RecognizerInstance] = None
        self.state: str = "disconnected"
        self.audio_buffer: bytearray = bytearray()
        self.silence_frames_count: int = 0
        self.frames_in_listening: int = 0
        self.llm_tts_task: Optional[asyncio.Task] = None
        self.is_websocket_active: bool = False

        self.local_audio_output_engine: Optional[AudioOutputEngineBase] = None

        # Состояния для адаптивного VAD и "залипания"
        self.estimated_noise_floor: float = self.vad_processing_settings.probability_threshold # Начальное предположение
        self.current_dynamic_vad_threshold: float = self.vad_processing_settings.probability_threshold
        self.potential_silence_frames: int = 0 # Счетчик для "залипания" речи
        self.was_recently_voiced: bool = False   # Флаг, была ли речь активна недавно

        if self.wake_word_engine and self.global_audio_settings.frame_length != self.wake_word_engine.frame_length:
            logger.warning(f"Global audio frame length ({self.global_audio_settings.frame_length}) "
                           f"differs from WakeWord frame length ({self.wake_word_engine.frame_length}). Ensure input matches WW.")
        if self.vad_engine and self.global_audio_settings.frame_length != self.vad_engine.frame_length:
            logger.warning(f"Global audio frame length ({self.global_audio_settings.frame_length}) "
                           f"differs from VAD frame length ({self.vad_engine.frame_length}).")
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
            await websocket.accept()
            await websocket.close(code=1008, reason="Server busy")
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
            logger.info("WebSocket connected, stopping any local audio output.")
            audio_output_engine.stop() 

        if not self.stt_provider:
            logger.error("STT provider not available!")
            await self.send_error("Server STT error.")
            await self.disconnect(code=1011)
            return False
        
        try:
            if not self.stt_recognizer: 
                self.stt_recognizer = self.stt_provider.create_recognizer()
                logger.info("STT recognizer created.")
            else: 
                self.stt_recognizer.reset()
                logger.info("STT recognizer reset.")
        except Exception as e:
            logger.error(f"Failed to create/reset STT recognizer: {e}", exc_info=True)
            await self.send_error("Server STT engine failure.")
            await self.disconnect(code=1011)
            return False
        
        # Check TTS engine health on connect as well
        if not self.tts_engine or not await self.tts_engine.is_healthy():
            logger.error("TTS engine not available or not healthy!")
            await self.send_error("Server TTS error.")
            # Don't necessarily disconnect, but log critical error.
            # Client might still be able to send audio for STT.
            # await self.disconnect(code=1011, reason="Server TTS unavailable")
            # return False


        logger.info("Client connected via WebSocket.")
        await self.send_status("wakeword_listening", "Waiting for wake word...")
        return True

    async def disconnect(self, code: int = 1000, reason: str = "Client disconnected"):
        global audio_input_engine, audio_output_engine 
        self.is_websocket_active = False 

        if self.llm_tts_task and not self.llm_tts_task.done():
            self.llm_tts_task.cancel()
            try: await self.llm_tts_task
            except asyncio.CancelledError: logger.info("LLM/TTS task cancelled on disconnect.")
            except Exception as e: logger.error(f"Error during LLM/TTS task cancellation: {e}")
        self.llm_tts_task = None

        if self.websocket:
            ws = self.websocket
            self.websocket = None
            if ws.client_state == WebSocketState.CONNECTED:
                try: await ws.close(code=code, reason=reason)
                except Exception as e: logger.warning(f"Error closing WebSocket: {e}")
        
        self.state = "disconnected" 

        logger.info(f"WebSocket client disconnected. Code: {code}, Reason: {reason}")

        if audio_input_engine and audio_input_engine.is_enabled:
            logger.info("WebSocket disconnected, resuming local audio input.")
            if not audio_input_engine.is_running: audio_input_engine.start()
            audio_input_engine.resume()
            self.state = "wakeword" 
            logger.info("Local audio interface active, waiting for wake word.")
        if audio_output_engine and audio_output_engine.is_enabled:
            if not audio_output_engine.is_running: audio_output_engine.start()


    async def _send_json(self, data: dict):
        if self.is_websocket_active and self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await self.websocket.send_json(data)
                return True
            except Exception as e:
                 logger.error(f"Error sending JSON to client: {e}")
                 return False
        return False

    async def send_status(self, status_code: str, message: str):
        await self._send_json({"type": "status", "code": status_code, "message": message})

    async def send_transcript(self, transcript: str, is_final: bool):
         await self._send_json({"type": "transcript", "text": transcript, "is_final": is_final})

    async def send_tts_chunk(self, audio_chunk_b64: str):
        if self.is_websocket_active:
            await self._send_json({"type": "tts_chunk", "data": audio_chunk_b64})

    async def send_tts_finished(self):
        if self.is_websocket_active:
            await self._send_json({"type": "tts_finished"})

    async def send_error(self, error_message: str):
        logger.error(f"Sending error to client: {error_message}")
        if self.is_websocket_active:
            await self._send_json({"type": "error", "message": error_message})
    
    async def _run_llm_tts_or_offline(self, text: str, thread_id: str):
            response_text = ""
            tts_audio_bytes_for_local = bytearray()
            current_tts_sample_rate_for_local_playback = None # Store the SR for local playback
            
            try:
                if self.is_websocket_active:
                    await self.send_status("processing_started", "Thinking...")
                else:
                    logger.info("Processing request (local audio)...")
                self.state = "processing"

                online_capable = await is_internet_available()
                attempt_online_processing = online_capable and settings.ai.online_mode and self.llm_logic_engine

                if attempt_online_processing:
                    logger.info("Using online LLM (LangGraph).")
                    lumi_response = await self.llm_logic_engine.ask(text, thread_id=thread_id) # type: ignore
                    response_text = lumi_response if isinstance(lumi_response, str) else lumi_response.content # type: ignore
                    if not response_text: response_text = "Sorry, I didn't get a response from the online assistant."

                else: 
                    logger.info(f"Offline processing path activated (Internet: {online_capable}, Online Mode: {settings.ai.online_mode}).")
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
                    else: 
                        if self.offline_llm_logic_engine:
                            logger.info("NLU did not identify a command / NLU unavailable. Trying offline LLM (Ollama).")
                            lumi_response = await self.offline_llm_logic_engine.ask(text, thread_id=thread_id)
                            response_text = lumi_response if isinstance(lumi_response, str) else lumi_response.content # type: ignore
                            if not response_text: response_text = "The offline assistant didn't provide a response."
                        else: 
                            logger.info("NLU did not identify a command, and offline LLM is not available.")
                            response_text = "I can only process specific commands offline, and this wasn't one of them. The general offline assistant is also unavailable."
                
                # TTS Synthesis
                if response_text and self.tts_engine: # self.tts_engine is Hybrid
                    active_tts_synthesizer = None
                    if isinstance(self.tts_engine, HybridTTSEngine):
                        active_tts_synthesizer = await self.tts_engine.get_active_engine_for_synthesis()
                    elif hasattr(self.tts_engine, 'get_output_sample_rate'): # Direct engine
                        active_tts_synthesizer = self.tts_engine
                    
                    if active_tts_synthesizer and await active_tts_synthesizer.is_healthy():
                        current_tts_sample_rate_for_local_playback = active_tts_synthesizer.get_output_sample_rate()
                        logger.info(f"TTS: Using {active_tts_synthesizer.__class__.__name__} with SR {current_tts_sample_rate_for_local_playback}Hz.")

                        if self.is_websocket_active:
                            await self.send_status("speaking_started", "Speaking...")
                        else:
                            logger.info(f"Synthesizing for local playback (SR: {current_tts_sample_rate_for_local_playback}Hz): '{response_text[:50]}...'")
                        self.state = "speaking"

                        async for tts_chunk in active_tts_synthesizer.synthesize_stream(response_text):
                            if self.is_websocket_active:
                                await self.send_tts_chunk(base64.b64encode(tts_chunk).decode('utf-8'))
                            if self.local_audio_output_engine and self.local_audio_output_engine.is_enabled:
                                tts_audio_bytes_for_local.extend(tts_chunk)
                        
                        if self.is_websocket_active: await self.send_tts_finished()
                        
                        if tts_audio_bytes_for_local and self.local_audio_output_engine and \
                        self.local_audio_output_engine.is_enabled and not self.is_websocket_active:
                            logger.info(f"Queueing {len(tts_audio_bytes_for_local)} bytes for local TTS (SR: {current_tts_sample_rate_for_local_playback}Hz).")
                            self.local_audio_output_engine.play_tts_bytes(
                                bytes(tts_audio_bytes_for_local),
                                sample_rate=current_tts_sample_rate_for_local_playback
                            )
                    else:
                        logger.error("TTS: No active and healthy synthesizer found or TTS engine is not correctly configured.")
                        if self.is_websocket_active: await self.send_error("Sorry, I can't speak right now (TTS synth error).")

                elif not response_text:
                    logger.warning("No response text generated by any LLM/NLU path.")
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

        bytes_per_sample = 2
        expected_bytes = self.global_audio_settings.frame_length * bytes_per_sample

        if len(audio_chunk) != expected_bytes:
            logger.warning(f"Audio chunk size mismatch: got {len(audio_chunk)}, expected {expected_bytes}. Skipping.")
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
                self.state = "listening"
                self.audio_buffer.clear()
                self.silence_frames_count = 0
                self.frames_in_listening = 0
                self.potential_silence_frames = 0 # Сброс для новой сессии слушания
                self.was_recently_voiced = False    # Сброс для новой сессии слушания
                # Сброс оценки шума до начального значения или порога по умолчанию при активации
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
                # Обновляем оценку уровня шума, если вероятность голоса низкая
                # (предполагаем, что это шум)
                if voice_probability < self.current_dynamic_vad_threshold * 0.7: # Эвристика
                    self.estimated_noise_floor = (1 - self.vad_processing_settings.noise_floor_alpha) * self.estimated_noise_floor + \
                                                 self.vad_processing_settings.noise_floor_alpha * voice_probability
                    self.estimated_noise_floor = max(0.0, min(self.estimated_noise_floor, 1.0)) # Ограничиваем 0-1

                # Рассчитываем динамический порог: немного выше оцененного шума
                self.current_dynamic_vad_threshold = self.estimated_noise_floor + self.vad_processing_settings.threshold_margin_factor
                self.current_dynamic_vad_threshold = max(self.vad_processing_settings.min_dynamic_threshold,
                                                         min(self.current_dynamic_vad_threshold, self.vad_processing_settings.max_dynamic_threshold))
                active_threshold = self.current_dynamic_vad_threshold
                # if self.frames_in_listening % 10 == 0: # Логирование для отладки
                #     logger.debug(f"VAD Prob: {voice_probability:.3f}, NoiseEst: {self.estimated_noise_floor:.3f}, DynThr: {active_threshold:.3f}")
            else:
                active_threshold = self.vad_processing_settings.probability_threshold

            is_voiced = voice_probability > active_threshold

            if is_voiced:
                self.silence_frames_count = 0 # Сбрасываем счетчик тишины, так как есть голос
                self.potential_silence_frames = 0 # Сбрасываем счетчик "потенциальной тишины"
                self.was_recently_voiced = True     # Отмечаем, что голос был активен
            else: # Голоса нет (is_voiced == False)
                if self.was_recently_voiced:
                    # Голос только что был, возможно, это короткая пауза ("залипание")
                    self.potential_silence_frames += 1
                    if self.potential_silence_frames > self.vad_processing_settings.speech_hangover_frames:
                        # Период "залипания" прошел, теперь это настоящая тишина
                        # Начинаем считать кадры реальной тишины с этого момента
                        self.silence_frames_count = self.potential_silence_frames - self.vad_processing_settings.speech_hangover_frames
                        # self.was_recently_voiced = False # Можно сбросить, но если голос снова появится, он снова станет True.
                                                        # Оставим True, чтобы следующая тишина тоже прошла через hangover.
                    # else: все еще в периоде "залипания", silence_frames_count остается 0
                else:
                    # Голоса не было недавно И сейчас тоже нет (продолжительная тишина после "залипания" или с самого начала)
                    self.silence_frames_count += 1

            grace_over = self.frames_in_listening >= self.vad_processing_settings.min_listening_frames
            # Используем self.silence_frames_count, который учитывает "залипание"
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
                        self.llm_tts_task.cancel()
                    self.llm_tts_task = asyncio.create_task(
                        self._run_llm_tts_or_offline(final_transcript, thread_id)
                    )
                else:
                    self.state = "wakeword"
                    if self.is_websocket_active:
                        await self.send_status("wakeword_listening", "No speech detected. Waiting for wake word...")
                    else:
                        logger.info("Local: No speech detected. Waiting for wake word...")

                self.audio_buffer.clear()
                self.silence_frames_count = 0
                self.frames_in_listening = 0
                self.potential_silence_frames = 0 # Сброс состояния "залипания"
                self.was_recently_voiced = False    # Сброс состояния "залипания"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global manager 
    if not manager:
        logger.error("ConnectionManager not initialized for WebSocket endpoint.")
        await websocket.accept()
        await websocket.close(code=1011, reason="Server not ready")
        return

    connected = await manager.connect(websocket)
    if not connected: return

    try:
        while True:
            message = await websocket.receive_json() 
            if message.get("type") == "audio_chunk":
                audio_b64 = message.get("data")
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        await manager.process_audio(audio_bytes, ASSISTANT_THREAD_ID, is_local_source=False)
                    except (base64.binascii.Error, TypeError) as e:
                        logger.error(f"Invalid base64 audio data: {e}")
                        if manager.is_websocket_active: await manager.send_error("Invalid audio data.")
                    except Exception as e:
                         logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
                         if manager.is_websocket_active: await manager.send_error("Server error processing audio.")
            
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: code={e.code}, reason='{e.reason}'")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}", exc_info=True)
        if manager and manager.is_websocket_active: 
            await manager.send_error(f"Unexpected WebSocket error: {str(e)}")
    finally:
        if manager: await manager.disconnect()


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
            logger.info(f"SoundDevice TTS Output Sample Rate: {settings.sounddevice.tts_output_sample_rate} Hz")
        except Exception as e:
            logger.warning(f"Could not list audio devices on startup: {e}")
    
    # Log Gemini specific info
    if settings.engines.tts_engine == "hybrid" or settings.engines.tts_engine == "gemini":
        logger.info(f"Gemini TTS configured with model: {settings.gemini_tts.model}, voice: {settings.gemini_tts.voice_name}")
        if not settings.gemini_tts.api_key:
            logger.warning("GEMINI_API_KEY is not set. Gemini TTS will not function.")
        else:
            logger.info("GEMINI_API_KEY is set.")


    uvicorn.run(app, host=settings.webapp.host, port=settings.webapp.port)