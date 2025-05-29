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
from engines.nlu.rasa_nlu import RasaNLUEngine
from engines.llm_logic.langgraph_llm import LangGraphLLMEngine
from engines.llm_logic.ollama_llm import OllamaLLMEngine # NEW
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
tts_engine: Optional[TTSEngineBase] = None
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

def create_engine_instance(engine_type: str, engine_name: str, global_settings: Settings):
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
        elif engine_type == "nlu":
            if engine_name == "rasa":
                # RasaNLUEngine needs settings.rasa_nlu
                return RasaNLUEngine(config=global_settings.rasa_nlu)
        elif engine_type == "llm_logic":
            if engine_name == "langgraph":
                # LangGraphLLMEngine needs settings.ai and settings.postgres
                return LangGraphLLMEngine(config={
                    "ai_settings": global_settings.ai,
                    "postgres_settings": global_settings.postgres
                })
        elif engine_type == "offline_llm_logic": # This is for the offline LLM
             if engine_name == "ollama":
                if global_settings.ollama.base_url and global_settings.ollama.model:
                    return OllamaLLMEngine(config={"ollama_settings": global_settings.ollama})
                else:
                    logger.warning("Ollama engine selected but base_url or model not configured. Skipping.")
                    return None
        elif engine_type == "audio_input":
            if engine_name == "sounddevice":
                # Needs callback, loop, and specific configs
                # Callback will be set after ConnectionManager is initialized
                return SoundDeviceInputEngine(
                    process_audio_callback=None, # Will be set later
                    loop=asyncio.get_running_loop(), # Assuming this is called within an async context
                    config={
                        "sounddevice_settings": global_settings.sounddevice,
                        "audio_settings": global_settings.audio # Global audio settings
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
    global llm_logic_engine, offline_llm_logic_engine # Added offline_llm_logic_engine
    global audio_input_engine, audio_output_engine, comm_service, manager
    global offline_command_processor

    logger.info("Initializing global engines...")

    # Communication Service (e.g., MQTT) - needed by some tools/NLU potentially
    comm_service = create_engine_instance("communication", app_settings.engines.communication_engine, app_settings)
    if comm_service:
        await comm_service.startup()
        # **Inject comm_service into tools module immediately after it's ready**
        from tools.device_control import initialize_device_control_tool # Ensure this import is here
        initialize_device_control_tool(comm_service)
        # You might need to do this for other tools if they also need comm_service
    else:
        logger.error("Communication service failed to initialize. Dependent services (offline processor, device tools) will be affected.")
        # No point in initializing tools that depend on a non-existent comm_service

    # 2. Initialize Offline Command Processor (needs comm_service)
    if comm_service:
        offline_command_processor = DefaultOfflineCommandProcessor(comm_service=comm_service)
        logger.info("DefaultOfflineCommandProcessor initialized.")
    else:
        offline_command_processor = None
        logger.warning("OfflineCommandProcessor not initialized as comm_service is unavailable.")

    # Core processing engines
    wake_word_engine = create_engine_instance("wake_word", app_settings.engines.wake_word_engine, app_settings)
    vad_engine = create_engine_instance("vad", app_settings.engines.vad_engine, app_settings)
    stt_engine = create_engine_instance("stt", app_settings.engines.stt_engine, app_settings)
    tts_engine = create_engine_instance("tts", app_settings.engines.tts_engine, app_settings)
    nlu_engine = create_engine_instance("nlu", app_settings.engines.nlu_engine, app_settings)
    # Online LLM
    llm_logic_engine = create_engine_instance("llm_logic", app_settings.engines.llm_logic_engine, app_settings)
    # Offline LLM
    offline_llm_logic_engine = create_engine_instance("offline_llm_logic", app_settings.engines.offline_llm_engine, app_settings)

    # Initialize ConnectionManager (needed for audio_input_engine callback)
    # Pass already initialized engines to ConnectionManager
    # Check primary engines; offline LLM is optional.
    if not all([wake_word_engine, vad_engine, stt_engine, tts_engine, nlu_engine, comm_service]): 
        logger.critical("One or more core processing or communication engines failed to initialize. Functionality will be severely limited.")
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
        llm_logic_engine=llm_logic_engine, # Online LLM
        offline_llm_logic_engine=offline_llm_logic_engine, # Offline LLM
        comm_service=comm_service,
        offline_processor=offline_command_processor,
        global_audio_settings=app_settings.audio,
        vad_processing_settings=app_settings.vad_config
    )
    logger.info("ConnectionManager initialized.")

    # Audio I/O engines (Local mic/speaker)
    # Audio Input
    _audio_input_engine_instance = create_engine_instance("audio_input", app_settings.engines.audio_input_engine, app_settings)
    if _audio_input_engine_instance and isinstance(_audio_input_engine_instance, AudioInputEngineBase):
        # Dynamically set the callback now that 'manager' exists
        async def process_audio_wrapper(audio_chunk: bytes):
            if manager: # Ensure manager is fully initialized
                await manager.process_audio(audio_chunk, ASSISTANT_THREAD_ID, is_local_source=True)
        _audio_input_engine_instance._process_audio_cb = process_audio_wrapper # Assuming direct access for simplicity
        audio_input_engine = _audio_input_engine_instance
        logger.info(f"AudioInputEngine ({app_settings.engines.audio_input_engine}) created.")
        if audio_input_engine and audio_input_engine.is_enabled:
             # Check compatibility with wake word engine
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


    # Audio Output (needs to be available for ConnectionManager)
    audio_output_engine = create_engine_instance("audio_output", app_settings.engines.audio_output_engine, app_settings)
    if audio_output_engine and audio_output_engine.is_enabled:
        logger.info(f"AudioOutputEngine ({app_settings.engines.audio_output_engine}) created and enabled.")
        # Pass to manager if it needs to directly control local playback (alternative to event bus)
        if manager: manager.set_local_audio_output(audio_output_engine)
    else:
        logger.info(f"AudioOutputEngine ({app_settings.engines.audio_output_engine}) is disabled or failed to initialize.")


    # Startup for engines that need it (TTS server, LLM DB connections, MQTT)
    # comm_service already started up
    if tts_engine: await tts_engine.startup()
    if llm_logic_engine: await llm_logic_engine.startup()
    # Audio I/O engines are typically started/stopped based on WebSocket connection or app state

    logger.info("Global engines initialization phase complete.")


async def shutdown_global_engines():
    logger.info("Shutting down global engines...")
    engine_list = [
        tts_engine, llm_logic_engine, offline_llm_logic_engine, comm_service, nlu_engine,
        wake_word_engine, vad_engine, stt_engine, 
        audio_input_engine, audio_output_engine 
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

    # Initialize scheduler database
    try:
        init_scheduler_db() # Call the DB init function from tools.scheduler
        init_music_likes_table() # <--- Для music_data.db (таблица liked_songs)
    except Exception as e:
        logger.error(f"Failed to initialize scheduler database: {e}. Reminders will not work.", exc_info=True)
        # Decide if this is a fatal error for your application

    await initialize_global_engines(settings) # Pass the global settings object

    # Start the reminder checker background task
    # It needs references to tts_engine and audio_output_engine
    global tts_engine, audio_output_engine # Ensure these are accessible
    if tts_engine and audio_output_engine:
         # Check if scheduler_db_path is set, otherwise reminder checker might not work well
        if not settings.scheduler_db_path:
            logger.warning("Scheduler DB path not set. Reminder checker might not function correctly.")
        await start_reminder_checker(tts_engine, audio_output_engine)
    else:
        logger.warning("TTS or Audio Output engine not available, reminder checker not started.")

    # Start local audio if enabled and no WebSocket is immediately active
    # This logic is now partly in ConnectionManager's connect/disconnect
    # and initial startup here.
    if audio_input_engine and audio_input_engine.is_enabled and \
       audio_output_engine and audio_output_engine.is_enabled and \
       manager and not manager.is_websocket_active:
        logger.info("Lifespan: No active WebSocket, starting local audio I/O.")
        audio_output_engine.start() # Start output capability
        audio_input_engine.start()  # Start microphone
        manager.state = "wakeword" # Set initial state for local audio
        logger.info("Lifespan: Local audio interface activated, waiting for wake word.")
    
    logger.info("Lifespan: Startup phase complete.")
    yield
    logger.info("Lifespan: Shutdown phase beginning...")
    await stop_reminder_checker() # Stop the reminder checker first
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
                 llm_logic_engine: Optional[LLMLogicEngineBase], # Online
                 offline_llm_logic_engine: Optional[LLMLogicEngineBase], # Offline
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
        self.offline_llm_logic_engine = offline_llm_logic_engine # Store it
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

        logger.info("WebSocket client disconnected.")

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
        
        try:
            if self.is_websocket_active:
                await self.send_status("processing_started", "Thinking...")
            else:
                 logger.info("Processing request (local audio)...")
            self.state = "processing"

            online = await is_internet_available()
            # Determine if online processing should be attempted
            attempt_online_processing = online and settings.ai.online_mode and self.llm_logic_engine

            if attempt_online_processing:
                logger.info("Using online LLM (LangGraph).")
                lumi_response = await self.llm_logic_engine.ask(text, thread_id=thread_id) # type: ignore
                response_text = lumi_response if isinstance(lumi_response, str) else lumi_response.content # type: ignore
                if not response_text: response_text = "Sorry, I didn't get a response from the online assistant."

            else: # Offline processing path
                logger.info("Offline processing path activated.")
                nlu_result = None
                if self.nlu_engine:
                    logger.debug("Attempting NLU parse for potential offline command.")
                    nlu_result = await self.nlu_engine.parse(text)

                if nlu_result and nlu_result.get("intent"): # NLU found a command
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
                    else: # NLU found command, but no processor
                        logger.warning("Offline NLU parsed a command, but no offline_processor is available.")
                        response_text = "I understood an offline command, but cannot process it right now."
                else: # NLU did not find a command (or NLU unavailable/failed) -> try offline LLM
                    if self.offline_llm_logic_engine:
                        logger.info("NLU did not identify a command / NLU unavailable. Trying offline LLM (Ollama).")
                        lumi_response = await self.offline_llm_logic_engine.ask(text, thread_id=thread_id)
                        response_text = lumi_response if isinstance(lumi_response, str) else lumi_response.content # type: ignore
                        if not response_text: response_text = "The offline assistant didn't provide a response."
                    else: # No NLU command AND no offline LLM
                        logger.info("NLU did not identify a command, and offline LLM is not available.")
                        response_text = "I can only process specific commands offline, and this wasn't one of them. The general offline assistant is also unavailable."
            
            # TTS Synthesis
            if response_text and self.tts_engine:
                if self.is_websocket_active:
                    await self.send_status("speaking_started", "Speaking...")
                else:
                    logger.info(f"Synthesizing for local playback: '{response_text[:50]}...'")
                self.state = "speaking"

                async for tts_chunk in self.tts_engine.synthesize_stream(response_text):
                    if self.is_websocket_active:
                        await self.send_tts_chunk(base64.b64encode(tts_chunk).decode('utf-8'))
                    if self.local_audio_output_engine and self.local_audio_output_engine.is_enabled:
                        tts_audio_bytes_for_local.extend(tts_chunk)
                
                if self.is_websocket_active: await self.send_tts_finished()
                
                if tts_audio_bytes_for_local and self.local_audio_output_engine and \
                   self.local_audio_output_engine.is_enabled and not self.is_websocket_active:
                    logger.info(f"Queueing {len(tts_audio_bytes_for_local)} bytes for local TTS.")
                    if not self.local_audio_output_engine.is_running:
                         self.local_audio_output_engine.start()
                    self.local_audio_output_engine.play_tts_bytes(bytes(tts_audio_bytes_for_local))
            
            elif not response_text:
                logger.warning("No response text generated by any LLM/NLU path.")
                # Potentially send a generic "I don't know" if WS is active
                if self.is_websocket_active:
                    await self.send_error("Sorry, I could not process your request.")


        except asyncio.CancelledError:
            logger.info("LLM/TTS/Offline background task cancelled.")
        except Exception as e:
            logger.error(f"Error in _run_llm_tts_or_offline: {e}", exc_info=True)
            if self.is_websocket_active: # Send error only if WS active
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
            if not is_local_source: await self.send_error("Server audio engines not ready.")
            return

        # Frame length check (use global audio settings for expected chunk structure)
        # WakeWord and VAD engines should match this frame_length from global_audio_settings
        # The input audio_chunk is raw bytes. Porcupine/Cobra need list of int16 samples.
        bytes_per_sample = 2 # int16
        expected_bytes = self.global_audio_settings.frame_length * bytes_per_sample
        
        if len(audio_chunk) != expected_bytes:
            logger.warning(f"Audio chunk size mismatch: got {len(audio_chunk)}, expected {expected_bytes}. Skipping.")
            return
        
        try:
            # Unpack bytes into list of int16 samples for Picovoice engines
            pcm = struct.unpack_from(f"{self.global_audio_settings.frame_length}h", audio_chunk)
        except struct.error as e:
            logger.error(f"Audio chunk unpack error: {e}. Length: {len(audio_chunk)}")
            return

        # --- State Machine ---
        if self.state == "wakeword":
            keyword_index = self.wake_word_engine.process(pcm)
            if keyword_index >= 0:
                logger.info("Wake word detected!")
                self.state = "listening"
                self.audio_buffer.clear()
                self.silence_frames_count = 0
                self.frames_in_listening = 0
                self.stt_recognizer.reset()
                if self.is_websocket_active:
                    await self.send_status("listening_started", "Listening...")
                else: # Local audio source
                     logger.info("Local: Listening started...")
                     # If local, and output was stopped, ensure it's ready for TTS
                     if self.local_audio_output_engine and not self.local_audio_output_engine.is_running:
                         self.local_audio_output_engine.start()


        elif self.state == "listening":
            self.frames_in_listening += 1
            self.audio_buffer.extend(audio_chunk) # STT engines might want full buffer later

            # STT processing
            self.stt_recognizer.accept_waveform(audio_chunk)
            partial_result_json = self.stt_recognizer.partial_result()
            partial_transcript = json.loads(partial_result_json).get("partial", "")
            if partial_transcript and self.is_websocket_active:
                await self.send_transcript(partial_transcript, is_final=False)

            # VAD processing
            voice_probability = self.vad_engine.process(pcm)
            is_voiced = voice_probability > self.vad_processing_settings.probability_threshold

            if not is_voiced:
                self.silence_frames_count += 1
            else:
                self.silence_frames_count = 0

            # End of speech conditions
            # Use self.vad_processing_settings for these thresholds
            grace_over = self.frames_in_listening >= self.vad_processing_settings.min_listening_frames
            silence_met = self.silence_frames_count >= self.vad_processing_settings.silence_frames_threshold
            max_len_met = self.frames_in_listening >= self.vad_processing_settings.max_listening_frames

            trigger_finalization = False
            if max_len_met:
                logger.info("Max listening frames reached, finalizing.")
                trigger_finalization = True
            elif grace_over and silence_met:
                logger.info("VAD detected end of speech.")
                trigger_finalization = True
            
            if trigger_finalization:
                final_result_json = self.stt_recognizer.final_result()
                final_transcript = json.loads(final_result_json).get("text", "").strip()
                logger.info(f"Final Transcript: '{final_transcript}'")

                if self.is_websocket_active:
                    await self.send_transcript(final_transcript, is_final=True)

                if final_transcript:
                    if self.llm_tts_task and not self.llm_tts_task.done():
                        self.llm_tts_task.cancel() # Cancel previous if any
                    self.llm_tts_task = asyncio.create_task(
                        self._run_llm_tts_or_offline(final_transcript, thread_id)
                    )
                    # State change to 'processing' will happen inside the task
                    # For responsiveness, we could set to 'wakeword' here, but the task manages it.
                else: # No transcript
                    self.state = "wakeword"
                    if self.is_websocket_active:
                        await self.send_status("wakeword_listening", "No speech detected. Waiting for wake word...")
                    else:
                        logger.info("Local: No speech detected. Waiting for wake word...")
                
                self.audio_buffer.clear()
                self.silence_frames_count = 0
                self.frames_in_listening = 0


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global manager # Access the global ConnectionManager instance
    if not manager:
        logger.error("ConnectionManager not initialized for WebSocket endpoint.")
        await websocket.accept()
        await websocket.close(code=1011, reason="Server not ready")
        return

    connected = await manager.connect(websocket)
    if not connected: return

    try:
        while True:
            message = await websocket.receive_json() # Assuming client sends JSON with audio
            if message.get("type") == "audio_chunk":
                audio_b64 = message.get("data")
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        await manager.process_audio(audio_bytes, ASSISTANT_THREAD_ID, is_local_source=False)
                    except (base64.binascii.Error, TypeError) as e:
                        logger.error(f"Invalid base64 audio data: {e}")
                        await manager.send_error("Invalid audio data.")
                    except Exception as e:
                         logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
                         await manager.send_error("Server error processing audio.")
            # Handle other message types if any (e.g., client config, stop commands)
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: code={e.code}, reason='{e.reason}'")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}", exc_info=True)
        if manager and manager.is_websocket_active: # Try to send error if still connected conceptually
            await manager.send_error(f"Unexpected WebSocket error: {str(e)}")
    finally:
        if manager: await manager.disconnect()


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Lumi Voice Assistant server (Modular) on {settings.webapp.host}:{settings.webapp.port}")
    
    # Log local audio device info if enabled
    if settings.sounddevice.enabled:
        try:
            import sounddevice as sd
            logger.info("Available Audio Devices (for SoundDevice engine):")
            logger.info(f"\n{sd.query_devices()}")
            logger.info(f"Using Input Device Index: {settings.sounddevice.input_device_index if settings.sounddevice.input_device_index is not None else 'Default'}")
            logger.info(f"Using Output Device Index: {settings.sounddevice.output_device_index if settings.sounddevice.output_device_index is not None else 'Default'}")
        except Exception as e:
            logger.warning(f"Could not list audio devices on startup: {e}")

    uvicorn.run(app, host=settings.webapp.host, port=settings.webapp.port)