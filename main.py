import asyncio
import base64
import json
import logging
import os
import struct
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# Picovoice
import pvporcupine
import pvcobra

# Vosk
from vosk import Model as VoskModel, KaldiRecognizer

# Project specific imports
from settings import settings
from llm.llm import ask_lumi 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Resources (Initialize Lazily or on Startup) ---
porcupine = None
cobra = None
vosk_model = None
paroli_server_process = None 
paroli_log_reader_task = None 

from functools import partial

# --- Paroli Server Management ---

async def read_stream(stream, log_prefix):
    """Асинхронно читает и логирует вывод из потока (stdout/stderr)."""
    while True:
        line = await stream.readline()
        if line:
            logger.info(f"[{log_prefix}] {line.decode('utf-8', errors='ignore').strip()}")
        else:
            break # Поток закрыт

async def start_paroli_server():
    """Запускает процесс paroli-server при старте FastAPI."""
    global paroli_server_process, paroli_log_reader_task
    if paroli_server_process is not None and paroli_server_process.returncode is None:
        logger.warning("Paroli server process already seems to be running.")
        return

    # Проверяем наличие исполняемого файла
    if not os.path.exists(settings.paroli_server.executable):
        logger.error(f"Paroli server executable not found at: {settings.paroli_server.executable}")
        logger.error("TTS will not be available. Please check settings.paroli_server.executable")
        # raise FileNotFoundError(f"Paroli server executable not found: {settings.paroli_server.executable}")
        return # Продолжаем без TTS

    # Формируем команду запуска
    command = [
        settings.paroli_server.executable,
        "--encoder", settings.paroli_server.encoder_path,
        "--decoder", settings.paroli_server.decoder_path,
        "-c", settings.paroli_server.config_path,
        "--ip", settings.paroli_server.ip,
        "--port", str(settings.paroli_server.port),
    ]
    # Добавляем дополнительные аргументы, если они есть
    command.extend(settings.paroli_server.extra_args)

    logger.info(f"Starting paroli-server with command: {' '.join(command)}")
    try:
        # Запускаем процесс, перенаправляя stdout и stderr
        paroli_server_process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        logger.info(f"Paroli server process started with PID: {paroli_server_process.pid}")

        # Запускаем задачи для асинхронного чтения логов
        stdout_task = asyncio.create_task(read_stream(paroli_server_process.stdout, "paroli-stdout"))
        stderr_task = asyncio.create_task(read_stream(paroli_server_process.stderr, "paroli-stderr"))
        paroli_log_reader_task = asyncio.gather(stdout_task, stderr_task) # Сохраняем для отмены

        # Даем серверу немного времени на запуск перед тем, как считать его готовым
        await asyncio.sleep(2.0) # Можно настроить время

        if paroli_server_process.returncode is not None:
             logger.error(f"Paroli server process exited immediately after start with code: {paroli_server_process.returncode}")
             paroli_server_process = None # Сбрасываем, так как он не запустился
             if paroli_log_reader_task:
                paroli_log_reader_task.cancel() # Отменяем задачи чтения логов
                paroli_log_reader_task = None
             # Можно поднять ошибку, чтобы FastAPI не стартовал
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
        paroli_log_reader_task.cancel() # Отменяем задачи чтения логов
        try:
            await paroli_log_reader_task # Ждем завершения отмены
        except asyncio.CancelledError:
            pass # Ожидаемое исключение
        paroli_log_reader_task = None
        logger.info("Paroli log reader tasks cancelled.")

    if paroli_server_process and paroli_server_process.returncode is None:
        logger.info(f"Stopping paroli-server process (PID: {paroli_server_process.pid})...")
        try:
            paroli_server_process.terminate() # Посылаем SIGTERM (более мягкий)
            await asyncio.wait_for(paroli_server_process.wait(), timeout=5.0) # Ждем до 5 секунд
            logger.info("Paroli server process terminated gracefully.")
        except asyncio.TimeoutError:
            logger.warning("Paroli server process did not terminate gracefully, killing...")
            try:
                paroli_server_process.kill() # Посылаем SIGKILL (жесткий)
                await paroli_server_process.wait() # Ждем завершения после kill
                logger.info("Paroli server process killed.")
            except ProcessLookupError:
                 logger.warning("Paroli server process already exited before kill.")
            except Exception as kill_err:
                logger.error(f"Error killing paroli-server process: {kill_err}")
        except ProcessLookupError:
             logger.warning("Paroli server process already exited before terminate.")
        except Exception as term_err:
             logger.error(f"Error terminating paroli-server process: {term_err}")
        finally:
            paroli_server_process = None
    else:
        logger.info("Paroli server process not running or already stopped.")


def initialize_resources():
    global porcupine, cobra, vosk_model
    try:
        porcupine = pvporcupine.create(
            access_key=settings.picovoice.access_key,
            keywords=["picovoice"],
        )
        logger.info(f"Porcupine initialized. Frame length: {porcupine.frame_length}, Sample rate: {porcupine.sample_rate}")
        if porcupine.sample_rate != settings.audio.sample_rate:
            logger.warning(f"Porcupine sample rate ({porcupine.sample_rate}) != configured rate ({settings.audio.sample_rate}). Resampling might be needed.")
        if porcupine.frame_length != settings.audio.frame_length:
             logger.warning(f"Porcupine frame length ({porcupine.frame_length}) != configured frame length ({settings.audio.frame_length}). Adjust settings.audio.frame_length.")

        # Cobra Voice Activity Detection
        cobra = pvcobra.create(
            access_key=settings.picovoice.access_key,
        )
        logger.info(f"Cobra VAD initialized. Frame length: {cobra.frame_length}, Sample rate: {cobra.sample_rate}")
        if cobra.sample_rate != settings.audio.sample_rate:
            logger.warning(f"Cobra sample rate ({cobra.sample_rate}) != configured rate ({settings.audio.sample_rate}). Resampling might be needed.")
        if cobra.frame_length != settings.audio.frame_length:
             logger.warning(f"Cobra frame length ({cobra.frame_length}) != configured frame length ({settings.audio.frame_length}). Adjust settings.audio.frame_length.")


        # Vosk Speech-to-Text
        if not os.path.exists(settings.vosk.model_path):
            raise FileNotFoundError(f"Vosk model not found at {settings.vosk.model_path}")
        vosk_model = VoskModel(settings.vosk.model_path)
        logger.info(f"Vosk model loaded from {settings.vosk.model_path}")


        # --- Paroli Server Sanity Check (проверка настроек) ---
        if not settings.paroli_server.executable:
             logger.warning("Path to paroli-server executable (paroli_server.executable) not set in settings.")
        if not settings.paroli_server.ws_url:
             logger.warning("Paroli Server WebSocket URL (paroli_server.ws_url) is missing or incorrect in settings.")
        if settings.paroli_server.audio_format == "pcm" and not settings.paroli_server.pcm_sample_rate:
             logger.warning("Paroli Server configured for 'pcm', but pcm_sample_rate not set. Playback may be incorrect.")
        logger.info(f"Paroli Server configured: Executable={settings.paroli_server.executable}, URL={settings.paroli_server.ws_url}, Format={settings.paroli_server.audio_format}")
        # --- End Paroli Server Sanity Check ---

    except pvporcupine.PorcupineError as e:
        logger.error(f"Porcupine initialization failed: {e}")
        raise
    except pvcobra.CobraError as e:
        logger.error(f"Cobra initialization failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error initializing resources: {e}")
        raise

# --- FastAPI App ---
app = FastAPI(
    title="Lumi Voice Assistant",
    on_startup=[initialize_resources, start_paroli_server],
    on_shutdown=[stop_paroli_server],                     
)

# Mount static files (HTML, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_root():
    """Serves the main HTML page."""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

# --- WebSocket Handler ---
class ConnectionManager:
    """Manages active WebSocket connections and their states."""
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.recognizers: dict[str, KaldiRecognizer] = {}
        self.states: dict[str, str] = {} # "wakeword", "listening", "processing", "speaking"
        self.audio_buffers: dict[str, bytearray] = {}
        self.silence_frames: dict[str, int] = {} # Count consecutive silent frames for VAD
        # Counter for frames processed since listening started
        self.frames_processed_in_listening: dict[str, int] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.states[client_id] = "wakeword" # Start by listening for wake word
        self.audio_buffers[client_id] = bytearray()
        self.silence_frames[client_id] = 0
        self.frames_processed_in_listening[client_id] = 0 # Initialize counter
        # Create a Vosk recognizer for this connection
        # Ensure vosk_model is initialized before this point
        if not vosk_model:
             logger.error("Vosk model not initialized during connect!")
             # Handle error appropriately, maybe close connection
             await websocket.close(code=1011, reason="Server configuration error")
             return
        self.recognizers[client_id] = KaldiRecognizer(vosk_model, settings.vosk.sample_rate)
        self.recognizers[client_id].SetWords(True) # Get word timings if needed
        logger.info(f"Client connected: {client_id}")
        await self.send_status(client_id, "wakeword_listening", "Waiting for wake word...")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.recognizers:
            del self.recognizers[client_id] # Clean up recognizer
        if client_id in self.states:
            del self.states[client_id]
        if client_id in self.audio_buffers:
            del self.audio_buffers[client_id]
        if client_id in self.silence_frames:
            del self.silence_frames[client_id]
        # Clean up frame counter
        if client_id in self.frames_processed_in_listening:
            del self.frames_processed_in_listening[client_id]
        logger.info(f"Client disconnected: {client_id}")

    async def send_status(self, client_id: str, status_code: str, message: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({"type": "status", "code": status_code, "message": message})
                except Exception as e:
                     logger.error(f"Error sending status to {client_id}: {e}")
                     # Maybe disconnect if sending fails repeatedly
                     # await self.disconnect_gracefully(websocket, client_id)


    async def send_transcript(self, client_id: str, transcript: str, is_final: bool):
         if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({
                        "type": "transcript",
                        "text": transcript,
                        "is_final": is_final
                    })
                except Exception as e:
                     logger.error(f"Error sending transcript to {client_id}: {e}")


    async def send_tts_chunk(self, client_id: str, audio_chunk_b64: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({
                        "type": "tts_chunk",
                        "data": audio_chunk_b64,
                        "is_final": False # Всегда False, т.к через tts_finished
                    })
                except Exception as e:
                     logger.error(f"Error sending TTS chunk to {client_id}: {e}")

    # --- Отправка сигнала об окончании потока TTS ---
    async def send_tts_finished(self, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({"type": "tts_finished"})
                    logger.info(f"Sent TTS finished signal to {client_id}")
                except Exception as e:
                     logger.error(f"Error sending TTS finished signal to {client_id}: {e}")

    async def send_error(self, client_id: str, error_message: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({"type": "error", "message": error_message})
                except Exception as e:
                     logger.error(f"Error sending error message to {client_id}: {e}")


    async def process_audio(self, client_id: str, audio_chunk: bytes):
            state = self.states.get(client_id)
            if not state:
                logger.warning(f"No state found for client {client_id}")
                return

            # --- Frame Length Check ---
            expected_bytes = settings.audio.frame_length * 2 # 2 bytes per int16 sample
            if len(audio_chunk) != expected_bytes:
                logger.warning(f"Received audio chunk size ({len(audio_chunk)} bytes) "
                                f"does not match expected ({expected_bytes} bytes). "
                                f"Engines might misbehave. Check client-side chunking.")
                # Decide handling... for now, proceed cautiously

            try:
                if len(audio_chunk) < expected_bytes:
                    logger.warning(f"Audio chunk too small ({len(audio_chunk)} bytes), expected {expected_bytes}. Skipping.")
                    return
                pcm = struct.unpack_from(f"{settings.audio.frame_length}h", audio_chunk)
            except struct.error as e:
                logger.error(f"Error unpacking audio chunk for {client_id}: {e}. Chunk length: {len(audio_chunk)}, Expected shorts: {settings.audio.frame_length}")
                return

            # --- State Machine ---
            if state == "wakeword":
                # --- Wake Word Logic ---
                try:
                    if len(pcm) != porcupine.frame_length:
                        logger.error(f"PCM length {len(pcm)} doesn't match Porcupine frame length {porcupine.frame_length}. Skipping wake word check.")
                    else:
                        keyword_index = porcupine.process(pcm)
                        if keyword_index >= 0:
                            logger.info(f"Wake word detected for {client_id}!")
                            self.states[client_id] = "listening"
                            self.audio_buffers[client_id] = bytearray()
                            self.silence_frames[client_id] = 0
                            self.frames_processed_in_listening[client_id] = 0
                            if client_id not in self.recognizers:
                                logger.error(f"Recognizer not found for {client_id} during wake word detection!")
                                self.recognizers[client_id] = KaldiRecognizer(vosk_model, settings.vosk.sample_rate)
                                self.recognizers[client_id].SetWords(True)
                            else:
                                self.recognizers[client_id].Reset()
                            await self.send_status(client_id, "listening_started", "Listening...")
                except pvporcupine.PorcupineError as e:
                    logger.error(f"Porcupine processing error for {client_id}: {e}")
                    await self.send_error(client_id, "Wake word engine error.")
                    self.states[client_id] = "wakeword"

            elif state == "listening":
                self.frames_processed_in_listening[client_id] += 1
                self.audio_buffers[client_id].extend(audio_chunk)
                vosk_recognizer = self.recognizers.get(client_id)
                if not vosk_recognizer:
                    logger.error(f"Recognizer not found for {client_id} in listening state!")
                    await self.send_error(client_id, "Internal error: Recognizer lost.")
                    self.states[client_id] = "wakeword"
                    return

                # --- Initialize variables for this chunk processing ---
                trigger_finalization = False # Flag to indicate VAD triggered end of speech
                # *** REMOVE intermediate final transcript variable ***
                # last_partial_transcript = "" # Optional: Store last partial if needed

                try:
                    # --- Process with Vosk ---
                    # Feed the chunk to Vosk
                    vosk_accepted_waveform = vosk_recognizer.AcceptWaveform(audio_chunk)

                    # Get partial result for continuous feedback
                    partial_result_json = vosk_recognizer.PartialResult()
                    partial_result = json.loads(partial_result_json)
                    partial_transcript = partial_result.get("partial", "")
                    # last_partial_transcript = partial_transcript # Store if needed for fallback
                    if partial_transcript:
                        # logger.debug(f"Vosk Partial: {partial_transcript}")
                        await self.send_transcript(client_id, partial_transcript, is_final=False)

                    # --- Process with Cobra VAD ---
                    is_voiced = False
                    if len(pcm) == cobra.frame_length:
                        voice_probability = cobra.process(pcm)
                        is_voiced = voice_probability > settings.vad.probability_threshold
                    else:
                        logger.warning(f"Audio chunk PCM size {len(pcm)} != Cobra frame length {cobra.frame_length}. VAD disabled for this frame.")
                        is_voiced = False # Assume silence if frame size is wrong

                    # --- VAD Logic ---
                    if not is_voiced:
                        self.silence_frames[client_id] += 1
                    else:
                        self.silence_frames[client_id] = 0 # Reset silence counter

                    grace_period_over = self.frames_processed_in_listening[client_id] >= settings.vad.min_listening_frames
                    silence_threshold_met = self.silence_frames[client_id] >= settings.vad.silence_frames

                    if grace_period_over and silence_threshold_met:
                        logger.info(f"VAD detected end of speech for {client_id} (Silence frames: {self.silence_frames[client_id]})")
                        trigger_finalization = True # Mark that VAD determined the end

                    # --- Finalization Logic (Triggered ONLY by VAD) ---
                    if trigger_finalization:
                        # Get the definitive final transcript using FinalResult()
                        # This should be called ONLY ONCE when VAD triggers.
                        final_result_json = vosk_recognizer.FinalResult()
                        final_result_data = json.loads(final_result_json)
                        final_transcript = final_result_data.get("text", "")

                        logger.debug(f"Vosk FinalResult JSON (after VAD): {final_result_json}")

                        # Log if the final transcript is empty
                        if not final_transcript:
                            logger.warning(f"Empty final transcript for {client_id} from FinalResult() after VAD trigger. Speech may have been too short, non-intelligible, or an issue with Vosk state.")
                            # Optional Fallback: If you stored the last partial transcript, you *could* use it here,
                            # but it's generally less reliable than FinalResult.
                            # if last_partial_transcript:
                            #    logger.warning(f"Using last known partial transcript as fallback: '{last_partial_transcript}'")
                            #    final_transcript = last_partial_transcript


                        logger.info(f"Final Transcript for {client_id} (from VAD trigger): '{final_transcript}'")
                        # Send the definitive final transcript (even if empty)
                        await self.send_transcript(client_id, final_transcript, is_final=True)

                        # --- Transition ---
                        if final_transcript:
                            self.states[client_id] = "processing"
                            await self.send_status(client_id, "processing_started", "Thinking...")
                            # (LLM and TTS call follows)
                            try:
                                loop = asyncio.get_running_loop()
                                func = partial(ask_lumi, final_transcript, thread_id=client_id)
                                lumi_response = await loop.run_in_executor(None, func)
                                logger.info(f"Lumi response for {client_id}: {lumi_response}")

                                self.states[client_id] = "speaking"
                                await self.send_status(client_id, "speaking_started", "Speaking...")
                                await self.synthesize_and_send_tts(client_id, lumi_response)

                            except Exception as e:
                                logger.error(f"Error during LLM call or TTS for {client_id}: {e}", exc_info=True)
                                await self.send_error(client_id, f"Error processing request: {e}")
                                if client_id in self.states:
                                    self.states[client_id] = "wakeword"
                                    await self.send_status(client_id, "wakeword_listening", "Waiting for wake word...")

                        else:
                            # No transcript, go back to wake word
                            logger.info(f"No final transcript obtained for {client_id}. Returning to wake word listening.")
                            self.states[client_id] = "wakeword"
                            await self.send_status(client_id, "wakeword_listening", "Waiting for wake word...")

                except pvcobra.CobraError as e:
                    logger.error(f"Cobra processing error for {client_id}: {e}")
                    await self.send_error(client_id, "VAD engine error.")
                    self.states[client_id] = "wakeword"
                except Exception as e:
                    logger.error(f"Error processing audio chunk for {client_id} in listening state: {e}", exc_info=True)
                    await self.send_error(client_id, f"Error processing audio: {e}")
                    self.states[client_id] = "wakeword"


    async def synthesize_and_send_tts(self, client_id: str, text: str):
        """Synthesizes text using Paroli Server WebSocket API and streams audio."""
        browser_websocket = self.active_connections.get(client_id)
        if not browser_websocket or browser_websocket.client_state != WebSocketState.CONNECTED:
            logger.warning(f"Client {client_id} disconnected before TTS could be sent.")
            return

        # Используем async with для автоматического управления соединением
        try:
            logger.info(f"Connecting to Paroli Server for TTS: {settings.paroli_server.ws_url}")
            async with websockets.connect(settings.paroli_server.ws_url) as paroli_ws:
                logger.info(f"Connected to Paroli Server for client {client_id}")

                # Construct request payload
                request_payload = {
                    "text": text,
                    "audio_format": settings.paroli_server.audio_format
                }
                # ... (добавление speaker_id и других опций как было) ...
                if settings.paroli_server.speaker_id is not None: request_payload["speaker_id"] = settings.paroli_server.speaker_id
                if settings.paroli_server.length_scale is not None: request_payload["length_scale"] = settings.paroli_server.length_scale
                if settings.paroli_server.noise_scale is not None: request_payload["noise_scale"] = settings.paroli_server.noise_scale
                if settings.paroli_server.noise_w is not None: request_payload["noise_w"] = settings.paroli_server.noise_w

                request_json = json.dumps(request_payload)
                logger.debug(f"Sending TTS request to Paroli Server for {client_id}: {request_json}")
                await paroli_ws.send(request_json)

                # Receive and forward audio chunks
                chunks_sent_to_browser = 0
                last_status_ok = False # Флаг для проверки успешного завершения
                while True:
                    # Check browser connection before receiving from Paroli
                    if client_id not in self.active_connections or self.active_connections[client_id].client_state != WebSocketState.CONNECTED:
                        logger.warning(f"Client {client_id} disconnected during TTS streaming from Paroli.")
                        break # Stop processing if browser client disconnected

                    try:
                        # Устанавливаем таймаут на получение, чтобы не зависнуть навсегда,
                        # если Paroli перестанет отвечать без закрытия соединения
                        message = await asyncio.wait_for(paroli_ws.recv(), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout waiting for message from Paroli Server for {client_id}.")
                        await self.send_error(client_id, "TTS server timed out.")
                        break
                    except websockets.exceptions.ConnectionClosedOK:
                        logger.info(f"Paroli Server closed connection cleanly for {client_id}.")
                        # Если соединение закрылось до получения "ok", считаем это ошибкой или незавершенным процессом
                        if not last_status_ok:
                            logger.warning(f"Paroli connection closed OK before receiving final 'ok' status for {client_id}.")
                            # Можно отправить ошибку или просто не отправлять tts_finished
                            await self.send_error(client_id, "TTS stream ended unexpectedly.")
                        break
                    except websockets.exceptions.ConnectionClosedError as e:
                        logger.error(f"Paroli Server connection closed unexpectedly for {client_id}: {e}")
                        await self.send_error(client_id, f"TTS Server connection error: {e.reason}")
                        break
                    except Exception as recv_err: # Ловим другие возможные ошибки при получении
                        logger.error(f"Error receiving message from Paroli Server for {client_id}: {recv_err}", exc_info=True)
                        await self.send_error(client_id, f"Error communicating with TTS server.")
                        break


                    if isinstance(message, bytes):
                        # Received binary audio chunk
                        if len(message) > 0:
                            audio_chunk_b64 = base64.b64encode(message).decode('utf-8')
                            await self.send_tts_chunk(client_id, audio_chunk_b64)
                            chunks_sent_to_browser += 1
                        else:
                            logger.warning(f"Received empty audio chunk from Paroli for {client_id}")

                    elif isinstance(message, str):
                        # Received final text status message
                        logger.info(f"Received final status from Paroli Server for {client_id}: {message}")
                        try:
                            status_data = json.loads(message)
                            if status_data.get("status") == "ok":
                                last_status_ok = True # Отмечаем успешное завершение
                            else:
                                error_msg = status_data.get("message", "Unknown TTS server error")
                                logger.error(f"Paroli Server TTS failed for {client_id}: {error_msg}")
                                await self.send_error(client_id, f"Text-to-Speech failed: {error_msg}")
                        except json.JSONDecodeError:
                            logger.error(f"Could not decode final status JSON from Paroli for {client_id}: {message}")
                            await self.send_error(client_id, "Received invalid final status from TTS server.")
                        break # End receiving loop after final text message
                    else:
                        logger.warning(f"Received unexpected message type from Paroli Server for {client_id}: {type(message)}")

                # После выхода из цикла while
                if last_status_ok: # Отправляем сигнал об окончании только если был статус "ok"
                     if client_id in self.active_connections and self.active_connections[client_id].client_state == WebSocketState.CONNECTED:
                         await self.send_tts_finished(client_id)
                     if chunks_sent_to_browser == 0:
                          logger.warning(f"Paroli finished OK but sent no audio chunks for {client_id}.")


                logger.info(f"Finished streaming TTS audio ({chunks_sent_to_browser} chunks) from Paroli Server for {client_id}")

            # Блок async with автоматически закроет соединение paroli_ws здесь,
            # даже если внутри цикла произошел break или исключение (кроме ConnectionClosed...)

        # Обработка ошибок подключения и других внешних ошибок
        except websockets.exceptions.InvalidURI:
            logger.error(f"Invalid Paroli Server WebSocket URL: {settings.paroli_server.ws_url}")
            await self.send_error(client_id, "Invalid TTS server address configured.")
        except ConnectionRefusedError:
             logger.error(f"Connection refused by Paroli Server at {settings.paroli_server.ws_url}. Is it running?")
             await self.send_error(client_id, "Could not connect to Text-to-Speech server.")
        except asyncio.TimeoutError: # Таймаут при подключении
             logger.error(f"Timeout connecting to Paroli Server at {settings.paroli_server.ws_url}.")
             await self.send_error(client_id, "Timeout connecting to Text-to-Speech server.")
        except Exception as e:
            logger.error(f"Paroli Server TTS Error for {client_id}: {e}", exc_info=True)
            if client_id in self.active_connections and self.active_connections[client_id].client_state == WebSocketState.CONNECTED:
                await self.send_error(client_id, f"Text-to-Speech error: {e}")
        finally:
            # Блок finally нужен только для перехода в состояние wakeword,
            # закрытие соединения обрабатывается через async with.
            if client_id in self.states and client_id in self.active_connections:
                 logger.info(f"TTS process finished for {client_id}, returning to wakeword state.")
                 self.states[client_id] = "wakeword"
                 browser_websocket = self.active_connections.get(client_id)
                 if browser_websocket and browser_websocket.client_state == WebSocketState.CONNECTED:
                     await self.send_status(client_id, "wakeword_listening", "Waiting for wake word...")
            else:
                 logger.info(f"Client {client_id} disconnected during or after Paroli TTS cleanup.")


manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Receive message - expecting JSON with base64 audio chunk
            message = await websocket.receive_json()
            if message.get("type") == "audio_chunk":
                audio_b64 = message.get("data")
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        # Process the audio chunk asynchronously
                        await manager.process_audio(client_id, audio_bytes)
                    except (base64.binascii.Error, TypeError) as e:
                        logger.error(f"Invalid base64 data received from {client_id}: {e}")
                        await manager.send_error(client_id, "Invalid audio data format.")
                    except Exception as e:
                         logger.error(f"Error processing message from {client_id}: {e}")
                         await manager.send_error(client_id, "Internal server error processing audio.")

            # Add handling for other message types if needed (e.g., client config)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket Error for client {client_id}: {e}")
        # Try to send an error before closing if possible
        if websocket.client_state == WebSocketState.CONNECTED:
             try:
                  await websocket.send_json({"type": "error", "message": f"WebSocket error: {e}"})
             except Exception:
                  pass # Ignore if sending fails during error handling
    finally:
        manager.disconnect(client_id)


# --- Run Server (for local development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.webapp.host}:{settings.webapp.port}")
    # Ensure models directory exists if specified relatively
    if not os.path.isabs(settings.vosk.model_path) and not os.path.exists("models"):
         os.makedirs("models/vosk", exist_ok=True)
         print("Created models/vosk directory. Please place Vosk model files there.")

    uvicorn.run(app, host=settings.webapp.host, port=settings.webapp.port)