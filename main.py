import asyncio
import base64
import json
import logging
import os
import struct
import wave
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# Picovoice
import pvporcupine
import pvcobra

# Vosk
from vosk import Model as VoskModel, KaldiRecognizer

# Piper TTS (using subprocess)
import subprocess
import tempfile

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

from functools import partial

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
    on_startup=[initialize_resources],
    # Add on_shutdown hook if needed to release resources, though OS usually handles it
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
        # NEW: Counter for frames processed since listening started
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
        # NEW: Clean up frame counter
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
                    await websocket.send_json({"type": "tts_chunk", "data": audio_chunk_b64})
                except Exception as e:
                     logger.error(f"Error sending TTS chunk to {client_id}: {e}")

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
                # --- Wake Word Logic (Same as before) ---
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

                    # *** We no longer call Result() here just because AcceptWaveform is True ***
                    # *** We only care about PartialResult for intermediate feedback ***

                    # Get partial result for continuous feedback
                    partial_result_json = vosk_recognizer.PartialResult()
                    partial_result = json.loads(partial_result_json)
                    partial_transcript = partial_result.get("partial", "")
                    # last_partial_transcript = partial_transcript # Store if needed for fallback
                    if partial_transcript:
                        # logger.debug(f"Vosk Partial: {partial_transcript}")
                        await self.send_transcript(client_id, partial_transcript, is_final=False)

                    # --- Process with Cobra VAD (Same as before) ---
                    is_voiced = False
                    if len(pcm) == cobra.frame_length:
                        voice_probability = cobra.process(pcm)
                        is_voiced = voice_probability > settings.vad.probability_threshold
                    else:
                        logger.warning(f"Audio chunk PCM size {len(pcm)} != Cobra frame length {cobra.frame_length}. VAD disabled for this frame.")
                        is_voiced = False # Assume silence if frame size is wrong

                    # --- VAD Logic (Same as before) ---
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

                        # --- Transition (Same as before) ---
                        if final_transcript:
                            self.states[client_id] = "processing"
                            await self.send_status(client_id, "processing_started", "Thinking...")
                            # (LLM and TTS call follows - same as before)
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
        """Synthesizes text using Piper TTS (via subprocess) and streams it."""
        websocket = self.active_connections.get(client_id)
        if not websocket or websocket.client_state != WebSocketState.CONNECTED:
            return # Client disconnected

        try:
            logger.info(f"Starting TTS synthesis for client {client_id}")
            # Use a temporary WAV file for output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                output_path = tmp_wav.name

            # Construct Piper command
            command = [
                settings.piper.executable,
                "--model", settings.piper.model_path,
                "--output_file", output_path,
            ]
            if settings.piper.config_path:
                command.extend(["--config", settings.piper.config_path])
            # Add other piper flags if needed (e.g., --length_scale, --noise_scale)

            logger.debug(f"Running Piper command: {' '.join(command)}")

            # Run Piper process, feeding text via stdin
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, # Capture stdout/stderr for debugging
                stderr=subprocess.PIPE
            )

            # Write text to piper's stdin and close it
            stdout, stderr = await process.communicate(input=text.encode('utf-8'))

            if process.returncode != 0:
                stderr_output = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Piper TTS failed with exit code {process.returncode}: {stderr_output}")
            else:
                 logger.info(f"Piper TTS synthesis successful. Output: {output_path}")
                 # Stream the generated WAV file back to the client
                 # Read the WAV file in chunks and send base64 encoded
                 chunk_size = 4096 # Send in 4KB chunks
                 with open(output_path, "rb") as wav_file:
                      while True:
                           chunk = wav_file.read(chunk_size)
                           if not chunk:
                                break
                           chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                           await self.send_tts_chunk(client_id, chunk_b64)
                           await asyncio.sleep(0.01) # Small sleep to prevent overwhelming the event loop/network

                 logger.info(f"Finished sending TTS audio for {client_id}")


        except FileNotFoundError:
             logger.error(f"Piper executable not found at '{settings.piper.executable}'. Please ensure it's installed and in PATH.")
             await self.send_error(client_id, "Text-to-Speech engine not found.")
        except Exception as e:
            logger.error(f"Piper TTS Error for {client_id}: {e}", exc_info=True)
            await self.send_error(client_id, f"Text-to-Speech error: {e}")
        finally:
            # Clean up temporary file
            if 'output_path' in locals() and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError as e:
                    logger.warning(f"Could not remove temporary TTS file {output_path}: {e}")

            # Regardless of TTS success/failure, transition back to wake word listening
            # Check if client still exists before updating state/sending status
            if client_id in self.states and client_id in self.active_connections:
                 logger.info(f"TTS finished for {client_id}, returning to wakeword state.")
                 self.states[client_id] = "wakeword"
                 await self.send_status(client_id, "wakeword_listening", "Waiting for wake word...")
            else:
                 logger.info(f"Client {client_id} disconnected during or after TTS.")



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
    if not os.path.isabs(settings.piper.model_path) and not os.path.exists("models"):
         os.makedirs("models/piper", exist_ok=True)
         print("Created models/piper directory. Please place Piper model files there.")

    uvicorn.run(app, host=settings.webapp.host, port=settings.webapp.port)