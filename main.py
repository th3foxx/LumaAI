# --- START OF FILE main.py ---

import asyncio
import base64
import json
import logging
import os
import struct
import websockets
import atexit # Для корректного закрытия ресурсов LLM

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
from mqtt_client import startup_mqtt_client, shutdown_mqtt_client, mqtt_client

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Resources ---
porcupine: pvporcupine.Porcupine | None = None
cobra: pvcobra.Cobra | None = None
vosk_model: VoskModel | None = None
paroli_server_process = None
paroli_log_reader_task = None

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


def initialize_resources():
    global porcupine, cobra, vosk_model
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

# --- FastAPI App ---
app = FastAPI(
    title="Lumi Voice Assistant Backend",
    on_startup=[
        initialize_resources,
        start_paroli_server,
        startup_mqtt_client
    ],
    on_shutdown=[
        stop_paroli_server,
        shutdown_mqtt_client,
        close_llm_resources # Закрываем ресурсы LLM при выходе
    ],
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

    async def connect(self, websocket: WebSocket):
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

        # Initialize Vosk recognizer for the connection
        if not vosk_model:
             logger.error("Vosk model not initialized! Cannot perform STT.")
             await self.send_error("Server configuration error: STT model not loaded.")
             await self.disconnect(code=1011) # Internal server error
             return False
        if not self.recognizer: # Create if it doesn't exist
            self.recognizer = KaldiRecognizer(vosk_model, settings.vosk.sample_rate)
            self.recognizer.SetWords(True) # Get word timings if needed
            logger.info("Vosk recognizer created.")
        else: # Reset if it exists from a previous connection
            self.recognizer.Reset()
            logger.info("Vosk recognizer reset.")

        logger.info("Client connected and ready.")
        await self.send_status("wakeword_listening", "Waiting for wake word...")
        return True

    async def disconnect(self, code: int = 1000, reason: str = "Client disconnected"):
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
        await self._send_json({
            "type": "tts_chunk",
            "data": audio_chunk_b64,
        })

    async def send_tts_finished(self):
        await self._send_json({"type": "tts_finished"})
        logger.info("Sent TTS finished signal to client.")

    async def send_error(self, error_message: str):
        logger.error(f"Sending error to client: {error_message}")
        await self._send_json({"type": "error", "message": error_message})

    async def _run_llm_and_tts(self, text: str, thread_id: str):
        """Runs LLM and TTS in the background."""
        logger.info(f"Starting background task: LLM query for '{text[:50]}...'")
        try:
            # 1. Send processing status
            await self.send_status("processing_started", "Thinking...")
            self.state = "processing" # Update state

            # 2. Call LLM (uses run_in_executor internally via LangGraph)
            # lumi_response = await asyncio.to_thread(ask_lumi, text, thread_id=thread_id) # Alternative if ask_lumi wasn't async friendly
            lumi_response = await ask_lumi(text, thread_id=thread_id)
            logger.info(f"LLM response received: '{lumi_response[:50]}...'")

            if not lumi_response:
                 logger.warning("LLM returned an empty response.")
                 await self.send_error("Assistant did not provide a response.")
                 # Go back to listening state directly
                 self.state = "wakeword"
                 await self.send_status("wakeword_listening", "Waiting for wake word...")
                 return # Exit background task

            # 3. Send speaking status and start TTS
            await self.send_status("speaking_started", "Speaking...")
            self.state = "speaking" # Update state
            await self.synthesize_and_send_tts(lumi_response) # This handles streaming TTS chunks

            # 4. TTS finished (or failed), transition back to wakeword
            # synthesize_and_send_tts now handles sending tts_finished
            # The 'finally' block in synthesize_and_send_tts is removed
            # We transition state *after* TTS completes or fails here.

        except asyncio.CancelledError:
             logger.info("LLM/TTS background task was cancelled.")
             # State might be processing or speaking, force back to wakeword if client still connected
             if self.state != "disconnected":
                self.state = "wakeword"
                await self.send_status("wakeword_listening", "Processing cancelled. Waiting for wake word...")
        except Exception as e:
            logger.error(f"Error during LLM call or TTS in background task: {e}", exc_info=True)
            await self.send_error(f"Error processing request: {str(e)}")
            # Go back to listening state after error
            if self.state != "disconnected":
                self.state = "wakeword"
                await self.send_status("wakeword_listening", "Error occurred. Waiting for wake word...")
        finally:
            # Ensure state is reset correctly if the task finishes normally or with non-critical errors handled within synthesize_and_send_tts
            if self.state not in ["disconnected", "wakeword"]: # If not already reset by error or cancellation
                 logger.info("LLM/TTS background task finished. Returning to wakeword state.")
                 self.state = "wakeword"
                 await self.send_status("wakeword_listening", "Waiting for wake word...")
            self.llm_tts_task = None # Clear the task reference


    async def process_audio(self, audio_chunk: bytes, thread_id: str):
            """Processes a single audio chunk based on the current state."""
            if self.state in ["processing", "speaking"]:
                # logger.debug("Ignoring audio chunk while processing or speaking.")
                return # Ignore audio while assistant is busy

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
                    keyword_index = porcupine.process(pcm)
                    if keyword_index >= 0:
                        logger.info("Wake word detected!")
                        self.state = "listening"
                        self.audio_buffer.clear()
                        self.silence_frames = 0
                        self.frames_processed_in_listening = 0
                        self.recognizer.Reset() # Reset Vosk for new utterance
                        await self.send_status("listening_started", "Listening...")
                except pvporcupine.PorcupineError as e:
                    logger.error(f"Porcupine processing error: {e}")
                    await self.send_error("Wake word engine error.")
                    # Remain in wakeword state

            elif self.state == "listening":
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
                    if partial_transcript:
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
                        await self.send_transcript(final_transcript, is_final=True)

                        # --- Trigger LLM/TTS or go back to Wakeword ---
                        if final_transcript:
                            # **** DECOUPLING ****
                            # Cancel any previous task just in case (shouldn't happen often)
                            if self.llm_tts_task and not self.llm_tts_task.done():
                                 logger.warning("Starting new LLM/TTS while previous one was still running. Cancelling old task.")
                                 self.llm_tts_task.cancel()

                            # Start LLM & TTS in background
                            self.state = "wakeword" # Set state immediately back to allow potential wake word detection later
                            self.llm_tts_task = asyncio.create_task(
                                self._run_llm_and_tts(final_transcript, thread_id)
                            )
                            # *DO NOT* await the task here. Let it run in the background.
                            # Status updates ("processing", "speaking") are handled within the task.
                            # For the user, it appears we are ready for the next command almost instantly.
                            logger.info("Scheduled LLM/TTS processing in background task.")
                            # No status sent here, _run_llm_and_tts sends "processing_started"

                        else:
                            # No transcript, just go back to wake word state
                            logger.info("Empty final transcript. Returning to wake word listening.")
                            self.state = "wakeword"
                            await self.send_status("wakeword_listening", "Waiting for wake word...")

                        # Clear buffers after finalization
                        self.audio_buffer.clear()
                        self.silence_frames = 0
                        self.frames_processed_in_listening = 0

                except Exception as e:
                    logger.error(f"Error processing audio chunk in listening state: {e}", exc_info=True)
                    await self.send_error(f"Error processing audio: {str(e)}")
                    self.state = "wakeword" # Reset state on error
                    await self.send_status("wakeword_listening", "Error occurred. Waiting for wake word...")
                    # Clear buffers on error too
                    self.audio_buffer.clear()
                    self.silence_frames = 0
                    self.frames_processed_in_listening = 0


    async def synthesize_and_send_tts(self, text: str):
        """Synthesizes text using Paroli Server and streams audio chunks."""
        if not self.websocket or self.websocket.client_state != WebSocketState.CONNECTED:
            logger.warning("Client disconnected before TTS could be sent.")
            return # Exit if client is gone

        if not settings.paroli_server.ws_url:
             logger.error("Paroli Server WebSocket URL not configured. Cannot synthesize TTS.")
             await self.send_error("TTS server is not configured.")
             return

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
                    # Check browser connection before receiving from Paroli
                    if not self.websocket or self.websocket.client_state != WebSocketState.CONNECTED:
                        logger.warning("Client disconnected during TTS streaming from Paroli.")
                        # Paroli connection will be closed by 'async with' automatically
                        return # Stop processing

                    try:
                        # Timeout for receiving data from Paroli
                        message = await asyncio.wait_for(paroli_ws.recv(), timeout=settings.paroli_server.receive_timeout) # Use setting
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for message from Paroli Server.")
                        await self.send_error("TTS server timed out.")
                        break # Exit loop, close Paroli connection
                    except websockets.exceptions.ConnectionClosedOK:
                        logger.info("Paroli Server closed connection cleanly.")
                        if not tts_success: # If closed OK before final 'ok' status
                             logger.warning("Paroli connection closed OK before receiving final 'ok' status.")
                             await self.send_error("TTS stream ended unexpectedly (OK).")
                        break # Exit loop
                    except websockets.exceptions.ConnectionClosedError as e:
                        logger.error(f"Paroli Server connection closed unexpectedly: {e}")
                        await self.send_error(f"TTS Server connection error: {e.reason}")
                        break # Exit loop
                    except Exception as recv_err:
                        logger.error(f"Error receiving message from Paroli Server: {recv_err}", exc_info=True)
                        await self.send_error("Error communicating with TTS server.")
                        break # Exit loop

                    if isinstance(message, bytes):
                        if len(message) > 0:
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
                                tts_success = True # Mark as successful
                            else:
                                error_msg = status_data.get("message", "Unknown TTS server error")
                                logger.error(f"Paroli Server TTS failed: {error_msg}")
                                await self.send_error(f"Text-to-Speech failed: {error_msg}")
                        except json.JSONDecodeError:
                            logger.error(f"Could not decode final status JSON from Paroli: {message}")
                            await self.send_error("Received invalid final status from TTS server.")
                        break # End receiving loop after final text message
                    else:
                        logger.warning(f"Received unexpected message type from Paroli Server: {type(message)}")

            # After 'async with' block (Paroli connection is closed)
            if tts_success:
                if chunks_sent_to_browser == 0:
                    logger.warning("Paroli finished OK but sent no audio chunks.")
                # Send the final signal *after* Paroli connection is closed and loop finishes
                await self.send_tts_finished()
            else:
                 logger.warning("TTS process finished without a success status from Paroli.")
                 # Error should have been sent already in the loop

            logger.info(f"Finished streaming TTS audio ({chunks_sent_to_browser} chunks). Success: {tts_success}")

        except websockets.exceptions.InvalidURI:
            logger.error(f"Invalid Paroli Server WebSocket URL: {settings.paroli_server.ws_url}")
            await self.send_error("Invalid TTS server address configured.")
        except ConnectionRefusedError:
             logger.error(f"Connection refused by Paroli Server at {settings.paroli_server.ws_url}. Is it running?")
             await self.send_error("Could not connect to Text-to-Speech server.")
        except asyncio.TimeoutError: # Connection timeout
             logger.error(f"Timeout connecting to Paroli Server at {settings.paroli_server.ws_url}.")
             await self.send_error("Timeout connecting to Text-to-Speech server.")
        except Exception as e:
            logger.error(f"Paroli Server TTS Error: {e}", exc_info=True)
            await self.send_error(f"Text-to-Speech error: {str(e)}")
        # finally:
            # **REMOVED**: State transition is now handled by the caller (_run_llm_and_tts)


manager = ConnectionManager()

# Use a fixed thread_id for the single assistant session
ASSISTANT_THREAD_ID = "lumi-voice-assistant-session"

@app.websocket("/ws") # Removed client_id from path
async def websocket_endpoint(websocket: WebSocket):
    """Handles the single WebSocket connection for the voice assistant."""
    connected = await manager.connect(websocket)
    if not connected:
        return # Connection was rejected or failed

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

    uvicorn.run(app, host=settings.webapp.host, port=settings.webapp.port)