# --- START OF FILE main.py ---

import asyncio
import base64
import json
import logging
import os
import struct
import threading
import websockets
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# Picovoice
import pvporcupine
import pvcobra

# Vosk
from vosk import Model as VoskModel, KaldiRecognizer

# Project specific imports
from settings import settings
from llm.llm import ask_lumi, close_llm_resources
from mqtt_client import startup_mqtt_client, shutdown_mqtt_client

from connectivity import is_internet_available
from offline_controller import parse_offline_command, execute_offline_command

from local_audio import LocalAudioManager

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Resources ---
porcupine: pvporcupine.Porcupine | None = None
cobra: pvcobra.Cobra | None = None
vosk_model: VoskModel | None = None
paroli_server_process = None
paroli_log_reader_task = None
local_audio_manager: LocalAudioManager | None = None # Add global ref

# --- Paroli Server Management ---

async def read_stream(stream, log_prefix):
    """Асинхронно читает и логирует вывод из потока (stdout/stderr)."""
    while True:
        try:
            line = await stream.readline()
            if line:
                logger.info(f"[{log_prefix}] {line.decode('utf-8', errors='ignore').strip()}")
            else:
                break # Поток закрыт
        except Exception as e:
            logger.error(f"Error reading stream {log_prefix}: {e}")
            break

async def start_paroli_server():
    """Запускает процесс paroli-server при старте FastAPI."""
    global paroli_server_process, paroli_log_reader_task
    if paroli_server_process is not None and paroli_server_process.returncode is None:
        logger.warning("Paroli server process already seems to be running.")
        return

    if not settings.paroli_server.executable or not os.path.exists(settings.paroli_server.executable):
        logger.error(f"Paroli server executable not found or not set: {settings.paroli_server.executable}")
        logger.error("TTS will not be available.")
        return # Продолжаем без TTS

    command = [
        settings.paroli_server.executable,
        "--encoder", settings.paroli_server.encoder_path,
        "--decoder", settings.paroli_server.decoder_path,
        "-c", settings.paroli_server.config_path,
        "--ip", settings.paroli_server.ip,
        "--port", str(settings.paroli_server.port),
    ]
    command.extend(settings.paroli_server.extra_args)

    logger.info(f"Starting paroli-server with command: {' '.join(command)}")
    try:
        paroli_server_process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        logger.info(f"Paroli server process started with PID: {paroli_server_process.pid}")

        stdout_task = asyncio.create_task(read_stream(paroli_server_process.stdout, "paroli-stdout"))
        stderr_task = asyncio.create_task(read_stream(paroli_server_process.stderr, "paroli-stderr"))
        paroli_log_reader_task = asyncio.gather(stdout_task, stderr_task)

        # Даем серверу немного времени на запуск
        await asyncio.sleep(3.0) # Увеличил немного

        if paroli_server_process.returncode is not None:
             logger.error(f"Paroli server process exited immediately after start with code: {paroli_server_process.returncode}")
             paroli_server_process = None
             if paroli_log_reader_task:
                paroli_log_reader_task.cancel()
                paroli_log_reader_task = None
             # Consider raising an error if TTS is critical
             # raise RuntimeError("Failed to start paroli-server")
        else:
             logger.info("Paroli server seems to be running.")

    except FileNotFoundError:
        logger.error(f"Paroli server executable not found at: {settings.paroli_server.executable}")
        paroli_server_process = None
    except Exception as e:
        logger.error(f"Failed to start paroli-server: {e}", exc_info=True)
        paroli_server_process = None

async def stop_paroli_server():
    """Останавливает процесс paroli-server при остановке FastAPI."""
    global paroli_server_process, paroli_log_reader_task
    if paroli_log_reader_task:
        paroli_log_reader_task.cancel()
        try:
            await paroli_log_reader_task
        except asyncio.CancelledError:
            pass
        paroli_log_reader_task = None
        logger.info("Paroli log reader tasks cancelled.")

    if paroli_server_process and paroli_server_process.returncode is None:
        pid = paroli_server_process.pid
        logger.info(f"Stopping paroli-server process (PID: {pid})...")
        try:
            paroli_server_process.terminate()
            await asyncio.wait_for(paroli_server_process.wait(), timeout=5.0)
            logger.info(f"Paroli server process (PID: {pid}) terminated gracefully.")
        except asyncio.TimeoutError:
            logger.warning(f"Paroli server process (PID: {pid}) did not terminate gracefully, killing...")
            try:
                paroli_server_process.kill()
                await paroli_server_process.wait()
                logger.info(f"Paroli server process (PID: {pid}) killed.")
            except ProcessLookupError:
                 logger.warning(f"Paroli server process (PID: {pid}) already exited before kill.")
            except Exception as kill_err:
                logger.error(f"Error killing paroli-server process (PID: {pid}): {kill_err}")
        except ProcessLookupError:
             logger.warning(f"Paroli server process (PID: {pid}) already exited before terminate.")
        except Exception as term_err:
             logger.error(f"Error terminating paroli-server process (PID: {pid}): {term_err}")
        finally:
            paroli_server_process = None
    else:
        logger.info("Paroli server process not running or already stopped.")


# --- Local Audio Shutdown Helper ---
async def shutdown_local_audio():
    """Shutdown local audio manager."""
    global local_audio_manager
    if local_audio_manager and local_audio_manager.is_enabled:
        logger.info("Shutting down local audio manager...")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, local_audio_manager.shutdown)
        logger.info("Local audio manager shutdown sequence initiated.")


def initialize_resources():
    global porcupine, cobra, vosk_model, local_audio_manager, manager # Ensure manager is global if accessed in lifespan startup
    logger.info("Initializing resources...")
    try:
        # Picovoice Porcupine (Wake Word)
        porcupine = pvporcupine.create(
            access_key=settings.picovoice.access_key,
            keywords=["picovoice"], # Or your custom keywords
            # model_path= Optional path
            # library_path= Optional path
        )
        logger.info(f"Porcupine initialized. Frame: {porcupine.frame_length}, Rate: {porcupine.sample_rate}, Version: {porcupine.version}")
        if porcupine.sample_rate != settings.audio.sample_rate:
            logger.warning(f"Porcupine sample rate ({porcupine.sample_rate}) != configured rate ({settings.audio.sample_rate}).")
        if porcupine.frame_length != settings.audio.frame_length:
             logger.warning(f"Porcupine frame length ({porcupine.frame_length}) != configured frame length ({settings.audio.frame_length}). Adjust settings.audio.frame_length.")

        # Picovoice Cobra (VAD)
        cobra = pvcobra.create(
            access_key=settings.picovoice.access_key,
            # library_path= Optional path
        )
        logger.info(f"Cobra VAD initialized. Frame: {cobra.frame_length}, Rate: {cobra.sample_rate}, Version: {cobra.version}")
        if cobra.sample_rate != settings.audio.sample_rate:
            logger.warning(f"Cobra sample rate ({cobra.sample_rate}) != configured rate ({settings.audio.sample_rate}).")
        if cobra.frame_length != settings.audio.frame_length:
             logger.warning(f"Cobra frame length ({cobra.frame_length}) != configured frame length ({settings.audio.frame_length}). Adjust settings.audio.frame_length.")

        # Vosk (STT)
        if not os.path.exists(settings.vosk.model_path):
            logger.error(f"Vosk model not found at {settings.vosk.model_path}. STT will not work.")
            # raise FileNotFoundError(f"Vosk model not found at {settings.vosk.model_path}") # Or allow startup without Vosk
            vosk_model = None # Explicitly set to None
        else:
            vosk_model = VoskModel(settings.vosk.model_path)
            logger.info(f"Vosk model loaded from {settings.vosk.model_path}. Sample rate expected: {settings.vosk.sample_rate}")
            if settings.vosk.sample_rate != settings.audio.sample_rate:
                 logger.warning(f"Vosk expected sample rate ({settings.vosk.sample_rate}) != configured rate ({settings.audio.sample_rate}). Recognition quality may suffer.")
        
        # --- Initialize Connection Manager FIRST ---
        # Needed by LocalAudioManager callback and lifespan startup logic
        manager = ConnectionManager() # Initialize manager here
        logger.info("ConnectionManager initialized.")

        # --- Initialize Local Audio Manager ---
        main_loop = asyncio.get_event_loop()
        async def process_audio_wrapper(audio_chunk: bytes):
             await manager.process_audio(audio_chunk, ASSISTANT_THREAD_ID)

        local_audio_manager = LocalAudioManager(process_audio_wrapper, main_loop)
        if local_audio_manager.is_enabled:
            logger.info("Local audio manager initialized.")
        else:
            logger.info("Local audio manager is disabled or failed to initialize.")


        # Paroli Server Sanity Check
        if not settings.paroli_server.executable:
             logger.warning("Path to paroli-server executable (paroli_server.executable) not set in settings.")
        if not settings.paroli_server.ws_url:
             logger.warning("Paroli Server WebSocket URL (paroli_server.ws_url) is missing or incorrect in settings.")
        if settings.paroli_server.audio_format == "pcm" and not settings.paroli_server.pcm_sample_rate:
             logger.warning("Paroli Server configured for 'pcm', but pcm_sample_rate not set. Playback may be incorrect.")
        logger.info(f"Paroli Server Config: Executable={settings.paroli_server.executable}, URL={settings.paroli_server.ws_url}, Format={settings.paroli_server.audio_format}")

        logger.info("Resource initialization complete.")

    except pvporcupine.PorcupineError as e:
        logger.error(f"Porcupine initialization failed: {e}", exc_info=True)
        raise # Critical failure
    except pvcobra.CobraError as e:
        logger.error(f"Cobra initialization failed: {e}", exc_info=True)
        raise # Critical failure
    except Exception as e:
        logger.error(f"Error initializing resources: {e}", exc_info=True)
        raise # Critical failure


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Startup Phase ===
    logger.info("Starting lifespan startup phase...")
    # 1. Initialize Core Resources (Non-async ok here)
    initialize_resources() # Initializes Vosk, Pico, LocalAudio, Manager

    # 2. Start Background Services (Async)
    await start_paroli_server()
    await startup_mqtt_client()

    # 3. Start Local Audio Interface (if needed) - Moved from old startup_event
    await asyncio.sleep(0.1) # Keep short delay
    global local_audio_manager, manager
    if local_audio_manager and local_audio_manager.is_enabled and not manager.is_websocket_active:
        logger.info("No active WebSocket connection on startup, starting local audio interface.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, local_audio_manager.start_output)
        await loop.run_in_executor(None, local_audio_manager.start_input)
        manager.state = "wakeword" # Set initial state for local interface
        logger.info("Local audio interface activated, waiting for wake word.")
    elif local_audio_manager and local_audio_manager.is_enabled and manager.is_websocket_active:
         logger.info("WebSocket connection already active on startup, local audio input remains paused.")

    logger.info("Lifespan startup phase complete.")

    yield # Application runs here

    # === Shutdown Phase ===
    logger.info("Starting lifespan shutdown phase...")
    await stop_paroli_server()
    await shutdown_mqtt_client()
    close_llm_resources()
    await shutdown_local_audio()
    logger.info("Lifespan shutdown phase complete.")


# --- FastAPI App ---
app = FastAPI(
    title="Lumi Voice Assistant Backend",
    lifespan=lifespan # <--- USE LIFESPAN INSTEAD OF on_startup/on_shutdown
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_root():
    """Serves the main HTML page."""
    from fastapi.responses import FileResponse
    # Ensure the file exists
    static_file_path = os.path.join("static", "index.html")
    if not os.path.exists(static_file_path):
        logger.error(f"Static file not found: {static_file_path}")
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(static_file_path)


# --- WebSocket Handler (Single Client Focus) ---
class ConnectionManager:
    """Manages the single active WebSocket connection and its state."""
    def __init__(self):
        self.websocket: WebSocket | None = None
        self.recognizer: KaldiRecognizer | None = None
        self.state: str = "disconnected" # "disconnected", "wakeword", "listening", "processing", "speaking"
        self.audio_buffer: bytearray = bytearray()
        self.silence_frames: int = 0
        self.frames_processed_in_listening: int = 0
        self.llm_tts_task: asyncio.Task | None = None # Task for background LLM/TTS
        self.is_websocket_active: bool = False # <<< ADDED FLAG

    async def connect(self, websocket: WebSocket):
        global local_audio_manager # Access global manager
        if self.websocket is not None and self.websocket.client_state == WebSocketState.CONNECTED:
            logger.warning("New connection attempt while another is active. Closing new connection.")
            await websocket.accept() # Accept then immediately close
            await websocket.close(code=1008, reason="Server busy with another client")
            return False

        await websocket.accept()
        self.websocket = websocket
        self.state = "wakeword"
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.frames_processed_in_listening = 0
        self.llm_tts_task = None # Ensure no lingering task
        self.is_websocket_active = True # <<< SET FLAG TO TRUE

         # Pause local audio input if it's running
        if local_audio_manager and local_audio_manager.is_enabled:
             logger.info("WebSocket connected, pausing local audio input.")
             # Run blocking pause in executor
             loop = asyncio.get_running_loop()
             await loop.run_in_executor(None, local_audio_manager.pause_input)
             # Also stop any potential local TTS playback immediately
             await loop.run_in_executor(None, local_audio_manager.stop_output)

        # Initialize Vosk recognizer for the connection
        if not vosk_model:
             logger.error("Vosk model not initialized! Cannot perform STT.")
             await self.send_error("Server configuration error: STT model not loaded.")
             await self.disconnect(code=1011) # Internal server error
             return False
        if not self.recognizer: # Create if it doesn't exist
            self.recognizer = KaldiRecognizer(vosk_model, settings.vosk.sample_rate)
            self.recognizer.SetWords(False) # Get word timings if needed
            logger.info("Vosk recognizer created.")
        else: # Reset if it exists from a previous connection
            self.recognizer.Reset()
            logger.info("Vosk recognizer reset.")

        logger.info("Client connected and ready.")
        await self.send_status("wakeword_listening", "Waiting for wake word...")
        return True

    async def disconnect(self, code: int = 1000, reason: str = "Client disconnected"):
        global local_audio_manager # Access global manager
        if self.llm_tts_task and not self.llm_tts_task.done():
            logger.warning("Client disconnected during LLM/TTS processing. Cancelling task.")
            self.llm_tts_task.cancel()
            try:
                await self.llm_tts_task # Allow cancellation to propagate
            except asyncio.CancelledError:
                logger.info("LLM/TTS task cancelled successfully.")
            except Exception as e:
                logger.error(f"Error during LLM/TTS task cancellation: {e}")
            self.llm_tts_task = None

        if self.websocket:
            ws = self.websocket # Temporary reference
            self.websocket = None # Mark as disconnected first
            if ws.client_state == WebSocketState.CONNECTED:
                try:
                    await ws.close(code=code, reason=reason)
                    logger.info(f"WebSocket connection closed gracefully (code={code}).")
                except Exception as e:
                    logger.warning(f"Error closing WebSocket: {e}")
            else:
                 logger.info(f"WebSocket already closed (state: {ws.client_state}).")

        # Don't delete recognizer, just reset it next time
        # if self.recognizer: del self.recognizer
        self.state = "disconnected"
        self.audio_buffer.clear()
        self.silence_frames = 0
        self.frames_processed_in_listening = 0
        logger.info("Client disconnected and resources cleaned.")

        # Resume local audio input if it exists and is enabled
        if local_audio_manager and local_audio_manager.is_enabled:
            logger.info("WebSocket disconnected, resuming local audio input.")
            # Run blocking resume/start in executor
            loop = asyncio.get_running_loop()
            # Ensure output is ready for potential local TTS later
            await loop.run_in_executor(None, local_audio_manager.start_output)
            # Resume input listening
            await loop.run_in_executor(None, local_audio_manager.resume_input)
            # If input wasn't running before, start it now
            if not local_audio_manager._is_input_running:
                 await loop.run_in_executor(None, local_audio_manager.start_input)
            # Optionally, reset state to wakeword for local interface
            self.state = "wakeword"
            logger.info("Local audio interface activated, waiting for wake word.")
            # Note: No status message sent as there's no WebSocket

    async def _send_json(self, data: dict):
        """Internal helper to send JSON safely."""
        if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await self.websocket.send_json(data)
                return True
            except Exception as e:
                 logger.error(f"Error sending JSON to client: {e} (Data: {str(data)[:100]}...)")
                 # Consider disconnecting if send fails repeatedly
                 # await self.disconnect(code=1011, reason="Communication error")
                 return False
        else:
            # logger.warning(f"Attempted to send JSON but client is disconnected. Data: {str(data)[:100]}...")
            return False

    async def send_status(self, status_code: str, message: str):
        await self._send_json({"type": "status", "code": status_code, "message": message})

    async def send_transcript(self, transcript: str, is_final: bool):
         await self._send_json({
             "type": "transcript",
             "text": transcript,
             "is_final": is_final
         })

    async def send_tts_chunk(self, audio_chunk_b64: str):
        """Sends TTS chunk to WebSocket if active."""
        if self.is_websocket_active:
            await self._send_json({
                "type": "tts_chunk",
                "data": audio_chunk_b64,
            })

    async def send_tts_finished(self):
        """Sends TTS finished signal to WebSocket if active."""
        if self.is_websocket_active:
            await self._send_json({"type": "tts_finished"})
            logger.info("Sent TTS finished signal to client (WebSocket).")

    async def send_error(self, error_message: str):
        """Sends error message to WebSocket if active."""
        logger.error(f"Sending error: {error_message}")
        if self.is_websocket_active:
            await self._send_json({"type": "error", "message": error_message})

    async def _run_llm_tts_or_offline(self, text: str, thread_id: str):
        """
        Runs LLM+TTS/Offline. Sends status/TTS to WebSocket OR plays TTS locally.
        """
        global local_audio_manager # Access global manager
        logger.info(f"Starting background task for text: '{text[:50]}...'")
        response_text = "" # Text to be spoken back to the user
        tts_audio_bytes = bytearray() # Buffer for local TTS playback
        was_online_request = False # Track if online resources were needed

        try:
            # 1. Send initial processing status (only if WS active)
            if self.is_websocket_active:
                await self.send_status("processing_started", "Thinking...")
            else:
                 logger.info("Processing request (local)...") # Log for local
            self.state = "processing" # Update state regardless of interface

            # 2. Check Internet Connectivity
            online = await is_internet_available()
            was_online_request = online and settings.ai.online_mode

            if online and settings.ai.online_mode:
                # --- ONLINE: Use LLM ---
                # ... (LLM logic - no changes) ...
                logger.info("Internet available. Using LLM.")
                lumi_response = await ask_lumi(text, thread_id=thread_id)
                logger.info(f"LLM response received: '{lumi_response[:50]}...'")
                if not lumi_response:
                     logger.warning("LLM returned an empty response.")
                     response_text = "Sorry, I didn't get a response from the assistant."
                else:
                     response_text = lumi_response
            else:
                # --- OFFLINE: Use Rasa NLU API or other logic ---
                # ... (Offline logic - no changes) ...
                logger.info("Internet unavailable or AI offline mode. Attempting offline command processing.")
                parsed_command = await parse_offline_command(text)
                if parsed_command:
                    logger.info(f"Offline command parsed via API: {parsed_command}")
                    execution_result = await execute_offline_command(parsed_command)
                    logger.info(f"Offline command execution result: {execution_result}")
                    response_text = execution_result
                else:
                    logger.warning(f"Offline command parsing via API failed for text: '{text}'")
                    response_text = "Sorry, I couldn't understand that command while offline."


            # 3. Synthesize Response (if any) and Send/Play
            if response_text:
                if self.is_websocket_active:
                    await self.send_status("speaking_started", "Speaking...")
                else:
                    logger.info("Synthesizing response for local playback...")
                self.state = "speaking" # Update state

                # Call synthesis, which now handles routing internally
                tts_audio_bytes = await self.synthesize_and_route_tts(response_text)

                # If local audio is active and we got bytes, play them
                if not self.is_websocket_active and local_audio_manager and local_audio_manager.is_enabled and tts_audio_bytes:
                    logger.info(f"Queueing {len(tts_audio_bytes)} bytes for local TTS playback.")
                    # Run blocking play call in executor
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, local_audio_manager.play_tts_bytes, tts_audio_bytes)
                    logger.info("Local TTS playback queued.")
                elif not self.is_websocket_active and not tts_audio_bytes:
                     logger.warning("TTS synthesis finished but produced no audio bytes for local playback.")

            else:
                 logger.warning("No response text generated to speak.")
                 # Go back to listening state directly (for either interface)
                 self.state = "wakeword"
                 if self.is_websocket_active:
                     await self.send_status("wakeword_listening", "Waiting for wake word...")
                 else:
                     logger.info("Waiting for wake word (local)...")


        except asyncio.CancelledError:
             logger.info("LLM/TTS/Offline background task was cancelled.")
             if self.state != "disconnected":
                self.state = "wakeword"
                if self.is_websocket_active:
                    await self.send_status("wakeword_listening", "Processing cancelled. Waiting for wake word...")
                else:
                    logger.info("Processing cancelled. Waiting for wake word (local)...")
        except Exception as e:
            logger.error(f"Error during LLM/TTS/Offline processing in background task: {e}", exc_info=True)
            await self.send_error(f"Error processing request: {str(e)}") # Sends only if WS active
            if self.state != "disconnected":
                self.state = "wakeword"
                if self.is_websocket_active:
                    await self.send_status("wakeword_listening", "Error occurred. Waiting for wake word...")
                else:
                    logger.info("Error occurred. Waiting for wake word (local)...")
        finally:
            # Ensure state is reset correctly
            if self.state not in ["disconnected", "wakeword"]:
                 logger.info("LLM/TTS/Offline background task finished. Returning to wakeword state.")
                 self.state = "wakeword"
                 if self.is_websocket_active:
                     await self.send_status("wakeword_listening", "Waiting for wake word...")
                 else:
                      logger.info("Waiting for wake word (local)...")
            self.llm_tts_task = None # Clear the task reference


    async def process_audio(self, audio_chunk: bytes, thread_id: str):
            """Processes a single audio chunk based on the current state."""
            if self.state in ["processing", "speaking"]:
                # logger.debug("Ignoring audio chunk while processing or speaking.")
                return # Ignore audio while assistant is busy
            
             # --- Ensure Recognizer Exists --- <--- ADD THIS BLOCK
            global vosk_model # Need access to the global model
            if not self.recognizer and vosk_model:
                 logger.info("Initializing Vosk recognizer instance...")
                 try:
                      # Ensure vosk_model is valid before creating recognizer
                      if vosk_model:
                           self.recognizer = KaldiRecognizer(vosk_model, settings.vosk.sample_rate)
                           self.recognizer.SetWords(False)
                           logger.info("Vosk recognizer instance created successfully.")
                      else:
                           # This case should ideally be caught during startup, but double-check
                           logger.error("Vosk model is None, cannot create recognizer instance.")
                           if self.is_websocket_active:
                                await self.send_error("Internal server error: STT model not loaded.")
                           return # Cannot proceed without a recognizer

                 except Exception as e:
                      logger.error(f"Failed to create Vosk recognizer instance: {e}", exc_info=True)
                      if self.is_websocket_active:
                           await self.send_error("Internal server error: STT engine failed to initialize.")
                      # If recognizer fails, we can't proceed
                      return

            if not self.recognizer or not porcupine or not cobra:
                logger.error("Required resources (Vosk, Porcupine, Cobra) not available.")
                await self.send_error("Internal server error: processing engines not ready.")
                # Consider disconnecting or attempting re-initialization
                return

            # --- Frame Length Check & Unpack ---
            expected_bytes = settings.audio.frame_length * 2 # 2 bytes per int16 sample
            if len(audio_chunk) != expected_bytes:
                # Pad or truncate? For now, log and skip if too small.
                if len(audio_chunk) < expected_bytes:
                    logger.warning(f"Received audio chunk too small ({len(audio_chunk)} bytes), expected {expected_bytes}. Skipping.")
                    return
                else:
                    logger.warning(f"Received audio chunk size ({len(audio_chunk)} bytes) differs from expected ({expected_bytes} bytes). Truncating.")
                    audio_chunk = audio_chunk[:expected_bytes]
                    # If larger, maybe process in loop? For now, truncate.

            try:
                pcm = struct.unpack_from(f"{settings.audio.frame_length}h", audio_chunk)
            except struct.error as e:
                logger.error(f"Error unpacking audio chunk: {e}. Chunk length: {len(audio_chunk)}, Expected shorts: {settings.audio.frame_length}")
                return

            # --- State Machine ---
            if self.state == "wakeword":
                # --- Wake Word Detection ---
                try:
                    pcm = struct.unpack_from(f"{settings.audio.frame_length}h", audio_chunk) # Unpack here
                    keyword_index = porcupine.process(pcm)
                    if keyword_index >= 0:
                        logger.info("Wake word detected!")
                        self.state = "listening"
                        self.audio_buffer.clear()
                        self.silence_frames = 0
                        self.frames_processed_in_listening = 0
                        if self.recognizer: self.recognizer.Reset() # Reset Vosk
                        if self.is_websocket_active:
                            await self.send_status("listening_started", "Listening...")
                        else:
                             logger.info("Listening (local)...") # Log for local interface
                except pvporcupine.PorcupineError as e:
                    logger.error(f"Porcupine processing error: {e}")
                    await self.send_error("Wake word engine error.")
                    # Remain in wakeword state
                except struct.error as e:
                     logger.error(f"Error unpacking audio chunk for Porcupine: {e}. Chunk length: {len(audio_chunk)}")
                     return # Can't proceed without unpacking

            elif self.state == "listening":
                try:
                    pcm = struct.unpack_from(f"{settings.audio.frame_length}h", audio_chunk) # Unpack for Cobra too
                except struct.error as e:
                    logger.error(f"Error unpacking audio chunk for VAD/STT: {e}. Chunk length: {len(audio_chunk)}")
                    # Maybe revert to wakeword state on unpacking error?
                    self.state = "wakeword"
                    if self.is_websocket_active: await self.send_status("wakeword_listening", "Audio error. Waiting for wake word...")
                    else: logger.info("Audio error. Waiting for wake word (local)...")
                    return
                
                self.frames_processed_in_listening += 1
                self.audio_buffer.extend(audio_chunk) # Keep buffering in case needed later

                # --- Initialize variables for this chunk processing ---
                trigger_finalization = False

                try:
                    # --- Process with Vosk STT ---
                    vosk_accepted_waveform = self.recognizer.AcceptWaveform(audio_chunk)

                    # Get partial result for feedback
                    partial_result_json = self.recognizer.PartialResult()
                    partial_result = json.loads(partial_result_json)
                    partial_transcript = partial_result.get("partial", "")
                    if partial_transcript and self.is_websocket_active: # Only send partial to WS
                        await self.send_transcript(partial_transcript, is_final=False)

                    # --- Process with Cobra VAD ---
                    is_voiced = False
                    try:
                        voice_probability = cobra.process(pcm)
                        is_voiced = voice_probability > settings.vad.probability_threshold
                    except pvcobra.CobraError as e:
                         logger.error(f"Cobra processing error: {e}")
                         await self.send_error("VAD engine error.")
                         self.state = "wakeword" # Revert state on VAD error
                         await self.send_status("wakeword_listening", "Error occurred. Waiting for wake word...")
                         if self.is_websocket_active: await self.send_status("wakeword_listening", "VAD Error. Waiting for wake word...")
                         else: logger.info("VAD Error. Waiting for wake word (local)...")
                         return # Stop processing this chunk

                    # --- VAD Logic ---
                    if not is_voiced:
                        self.silence_frames += 1
                    else:
                        self.silence_frames = 0 # Reset silence counter on voice

                    # Check conditions for ending speech
                    grace_period_over = self.frames_processed_in_listening >= settings.vad.min_listening_frames
                    silence_threshold_met = self.silence_frames >= settings.vad.silence_frames
                    max_length_reached = self.frames_processed_in_listening >= settings.vad.max_listening_frames # Add max length failsafe

                    if max_length_reached:
                         logger.warning(f"Max listening frames ({settings.vad.max_listening_frames}) reached. Forcing finalization.")
                         trigger_finalization = True
                    elif grace_period_over and silence_threshold_met:
                        logger.info(f"VAD detected end of speech (Silence frames: {self.silence_frames})")
                        trigger_finalization = True

                    # --- Finalization Logic (Triggered by VAD or Max Length) ---
                    if trigger_finalization:
                        final_result_json = self.recognizer.FinalResult()
                        final_result_data = json.loads(final_result_json)
                        final_transcript = final_result_data.get("text", "").strip()

                        logger.debug(f"Vosk FinalResult JSON: {final_result_json}")
                        logger.info(f"Final Transcript: '{final_transcript}'")

                        # Send final transcript (even if empty, client might want to know)
                        if self.is_websocket_active: # Send final transcript only to WS
                            await self.send_transcript(final_transcript, is_final=True)

                        # --- Trigger LLM/TTS or go back to Wakeword ---
                        if final_transcript:
                            if self.llm_tts_task and not self.llm_tts_task.done():
                                 logger.warning("Starting new processing while previous one was still running. Cancelling old task.")
                                 self.llm_tts_task.cancel()

                            # State change now happens inside _run_llm_tts_or_offline
                            # Start background task
                            self.llm_tts_task = asyncio.create_task(
                                self._run_llm_tts_or_offline(final_transcript, thread_id)
                            )
                            logger.info("Scheduled LLM/TTS/Offline processing in background task.")
                            # *Immediately* go back to wakeword state for responsiveness
                            self.state = "wakeword"
                            # Don't send status here, background task handles it

                        else:
                            # No transcript, just go back to wake word state
                            logger.info("Empty final transcript. Returning to wake word listening.")
                            self.state = "wakeword"
                            if self.is_websocket_active:
                                await self.send_status("wakeword_listening", "Waiting for wake word...")
                            else:
                                logger.info("Waiting for wake word (local)...")

                        # Clear buffers
                        self.audio_buffer.clear()
                        self.silence_frames = 0
                        self.frames_processed_in_listening = 0

                except Exception as e:
                    logger.error(f"Error processing audio chunk in listening state: {e}", exc_info=True)
                    await self.send_error(f"Error processing audio: {str(e)}")
                    self.state = "wakeword" # Reset state on error
                    if self.is_websocket_active: await self.send_status("wakeword_listening", "Error occurred. Waiting for wake word...")
                    else: logger.info("Error occurred. Waiting for wake word (local)...")
                    # Clear buffers on error too
                    self.audio_buffer.clear()
                    self.silence_frames = 0
                    self.frames_processed_in_listening = 0


    async def synthesize_and_route_tts(self, text: str) -> bytes:
        """Synthesizes text using Paroli Server and streams audio chunks."""
        global local_audio_manager
        all_tts_bytes = bytearray() # Accumulate bytes for local playback

        is_initially_websocket_active = self.is_websocket_active and self.websocket and self.websocket.client_state == WebSocketState.CONNECTED


        if not settings.paroli_server.ws_url:
             logger.error("Paroli Server WebSocket URL not configured. Cannot synthesize TTS.")
             if is_initially_websocket_active: # Send error only if WS was active initially
                 await self.send_error("TTS server is not configured.")
             return all_tts_bytes # Return empty

        tts_success = False
        chunks_sent_to_browser = 0
        try:
            logger.info(f"Connecting to Paroli TTS: {settings.paroli_server.ws_url}")
            async with websockets.connect(settings.paroli_server.ws_url, open_timeout=5.0, close_timeout=5.0) as paroli_ws:
                logger.info(f"Connected to Paroli TTS for synthesis.")

                request_payload = {
                    "text": text,
                    "audio_format": settings.paroli_server.audio_format
                }
                if settings.paroli_server.speaker_id is not None: request_payload["speaker_id"] = settings.paroli_server.speaker_id
                if settings.paroli_server.length_scale is not None: request_payload["length_scale"] = settings.paroli_server.length_scale
                if settings.paroli_server.noise_scale is not None: request_payload["noise_scale"] = settings.paroli_server.noise_scale
                if settings.paroli_server.noise_w is not None: request_payload["noise_w"] = settings.paroli_server.noise_w

                request_json = json.dumps(request_payload)
                logger.debug(f"Sending TTS request to Paroli: {request_json[:100]}...")
                await paroli_ws.send(request_json)

                # Receive and forward audio chunks
                while True:
                    # Check WebSocket status *inside* the loop for sending chunks
                    is_websocket_still_active_for_sending = self.is_websocket_active and self.websocket and self.websocket.client_state == WebSocketState.CONNECTED

                    message = None
                    try:
                        message = await asyncio.wait_for(paroli_ws.recv(), timeout=settings.paroli_server.receive_timeout)
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for message from Paroli Server.")
                        if is_websocket_still_active_for_sending: await self.send_error("TTS server timed out.")
                        break
                    except websockets.exceptions.ConnectionClosedOK:
                        logger.info("Paroli Server closed connection cleanly.")
                        if not tts_success and is_websocket_still_active_for_sending:
                             logger.warning("Paroli connection closed OK before receiving final 'ok' status.")
                             await self.send_error("TTS stream ended unexpectedly (OK).")
                        break
                    except websockets.exceptions.ConnectionClosedError as e:
                        logger.error(f"Paroli Server connection closed unexpectedly: {e}")
                        if is_websocket_still_active_for_sending: await self.send_error(f"TTS Server connection error: {e.reason}")
                        break
                    except Exception as recv_err:
                        logger.error(f"Error receiving message from Paroli Server: {recv_err}", exc_info=True)
                        if is_websocket_still_active_for_sending: await self.send_error("Error communicating with TTS server.")
                        break

                    if isinstance(message, bytes):
                        if len(message) > 0:
                            all_tts_bytes.extend(message) # Always accumulate
                            # Send only if WS is still active for sending
                            if is_websocket_still_active_for_sending:
                                audio_chunk_b64 = base64.b64encode(message).decode('utf-8')
                                await self.send_tts_chunk(audio_chunk_b64)
                                chunks_sent_to_browser += 1
                        else:
                            logger.warning("Received empty audio chunk from Paroli.")
                    elif isinstance(message, str):
                         logger.info(f"Received final status from Paroli Server: {message}")
                         try:
                              status_data = json.loads(message)
                              if status_data.get("status") == "ok":
                                   tts_success = True
                              else:
                                   error_msg = status_data.get("message", "Unknown TTS server error")
                                   logger.error(f"Paroli Server TTS failed: {error_msg}")
                                   if is_websocket_still_active_for_sending: await self.send_error(f"Text-to-Speech failed: {error_msg}")
                         except json.JSONDecodeError:
                              logger.error(f"Could not decode final status JSON from Paroli: {message}")
                              if is_websocket_still_active_for_sending: await self.send_error("Received invalid final status from TTS server.")
                         break # End receiving loop
                    else:
                        logger.warning(f"Received unexpected message type from Paroli Server: {type(message)}")

            # After Paroli connection is closed
            # Check final WS state for sending the finished signal
            is_websocket_still_active_for_finish_signal = self.is_websocket_active and self.websocket and self.websocket.client_state == WebSocketState.CONNECTED
            if tts_success:
                 if is_websocket_still_active_for_finish_signal and chunks_sent_to_browser == 0 and len(all_tts_bytes) > 0:
                     logger.warning("Paroli finished OK but sent no chunks to browser (maybe disconnected during?).")
                 elif is_websocket_still_active_for_finish_signal and len(all_tts_bytes) > 0:
                      await self.send_tts_finished() # Send finished signal
                 # Check original state for local playback warning
                 elif not is_initially_websocket_active and len(all_tts_bytes) == 0:
                      logger.warning("Paroli finished OK but produced no audio bytes for local playback.")
            else:
                  logger.warning("TTS process finished without a success status from Paroli.")

            logger.info(f"Finished TTS processing. Success: {tts_success}. Bytes generated: {len(all_tts_bytes)}. Chunks sent to WS: {chunks_sent_to_browser}")

        except websockets.exceptions.InvalidURI:
             logger.error(f"Invalid Paroli Server WebSocket URL: {settings.paroli_server.ws_url}")
             if is_initially_websocket_active: await self.send_error("Invalid TTS server address configured.")
        except ConnectionRefusedError:
             logger.error(f"Connection refused by Paroli Server at {settings.paroli_server.ws_url}. Is it running?")
             if is_initially_websocket_active: await self.send_error("Could not connect to Text-to-Speech server.")
        except asyncio.TimeoutError: # Connection timeout connecting to Paroli
             logger.error(f"Timeout connecting to Paroli Server at {settings.paroli_server.ws_url}.")
             if is_initially_websocket_active: await self.send_error("Timeout connecting to Text-to-Speech server.")
        except Exception as e:
            logger.error(f"Paroli Server TTS Error: {e}", exc_info=True)
             # Send error only if WS was active when the error likely occurred
            if is_initially_websocket_active:
                 await self.send_error(f"Text-to-Speech error: {str(e)}")
        finally:
            # Always return the accumulated bytes
            return bytes(all_tts_bytes)



# Use a fixed thread_id for the single assistant session
ASSISTANT_THREAD_ID = "lumi-voice-assistant-session"

@app.websocket("/ws") # Removed client_id from path
async def websocket_endpoint(websocket: WebSocket):
    """Handles the single WebSocket connection for the voice assistant."""
    global manager
    if not manager: # Check manager is initialized
        logger.error("Connection Manager not initialized before WebSocket connection attempt.")
        # Accept and close gracefully if manager isn't ready
        try:
            await websocket.accept()
            await websocket.close(code=1011, reason="Server not ready")
        except Exception as e:
             logger.error(f"Error accepting/closing websocket when manager not ready: {e}")
        return

    connected = await manager.connect(websocket)
    if not connected:
        return

    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "audio_chunk":
                audio_b64 = message.get("data")
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        # Process using the fixed thread ID
                        await manager.process_audio(audio_bytes, ASSISTANT_THREAD_ID)
                    except (base64.binascii.Error, TypeError) as e:
                        logger.error(f"Invalid base64 data received: {e}")
                        await manager.send_error("Invalid audio data format.")
                    except Exception as e:
                         logger.error(f"Error processing message: {e}", exc_info=True)
                         await manager.send_error("Internal server error processing audio.")
                else:
                    logger.warning("Received audio_chunk message with no data.")
            else:
                logger.warning(f"Received unknown message type: {message.get('type')}")

    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: code={e.code}, reason='{e.reason}'")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}", exc_info=True)
        # Try to send error before disconnecting if possible
        await manager.send_error(f"Unexpected WebSocket error: {str(e)}")
    finally:
        if manager: # Check if manager exists before calling disconnect
            await manager.disconnect()


# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Lumi Voice Assistant server on {settings.webapp.host}:{settings.webapp.port}")
    # Ensure Vosk model directory exists if relative path used
    vosk_dir = os.path.dirname(settings.vosk.model_path)
    if vosk_dir and not os.path.isabs(vosk_dir) and not os.path.exists(vosk_dir):
         try:
            os.makedirs(vosk_dir, exist_ok=True)
            logger.info(f"Created directory for Vosk model: {vosk_dir}")
         except OSError as e:
            logger.error(f"Could not create Vosk model directory {vosk_dir}: {e}")
    
    logger.info(f"Local audio enabled via settings: {settings.local_audio.enabled}")
    # List devices at startup for easier debugging
    if settings.local_audio.enabled:
        try:
            import sounddevice as sd
            logger.info("Available Audio Devices:")
            logger.info(f"\n{sd.query_devices()}")
            logger.info(f"Using Input Device Index: {settings.local_audio.input_device_index if settings.local_audio.input_device_index is not None else 'Default'}")
            logger.info(f"Using Output Device Index: {settings.local_audio.output_device_index if settings.local_audio.output_device_index is not None else 'Default'}")
        except Exception as e:
            logger.warning(f"Could not list audio devices on startup: {e}")

    uvicorn.run(app, host=settings.webapp.host, port=settings.webapp.port)