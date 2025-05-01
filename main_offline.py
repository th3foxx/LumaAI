import asyncio
import base64
import json
import logging
import os
import struct
import time
import queue # Use standard queue for thread-safe audio passing
import numpy as np
import sounddevice as sd
import websockets # For Paroli TTS client

# Picovoice
import pvporcupine
import pvcobra

# Vosk
from vosk import Model as VoskModel, KaldiRecognizer

# Project specific imports
from settings import settings
from llm.llm import ask_lumi # Assuming ask_lumi is synchronous or handled elsewhere

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Global Resources ---
porcupine = None
cobra = None
vosk_model = None
vosk_recognizer = None # Single recognizer now
paroli_server_process = None
paroli_log_reader_task = None
audio_queue = asyncio.Queue() # Queue for audio frames from input callback
tts_queue = queue.Queue() # Queue for TTS audio chunks to output thread

# --- Audio Settings (Derived from settings.py and engine requirements) ---
# Ensure these match your hardware and engine capabilities
SAMPLE_RATE = settings.audio.sample_rate
FRAME_LENGTH = settings.audio.frame_length # Use the setting directly
CHANNELS = 1
DTYPE = 'int16' # Corresponds to 'h' in struct packing

# --- State Management ---
class AssistantState:
    WAITING_FOR_WAKEWORD = 1
    LISTENING = 2
    PROCESSING_LLM = 3
    SPEAKING = 4

current_state = AssistantState.WAITING_FOR_WAKEWORD
silence_frames_count = 0
listening_frames_count = 0

# --- Paroli Server Management (Keep as is, slightly adapted) ---
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
    """Запускает процесс paroli-server."""
    global paroli_server_process, paroli_log_reader_task
    if paroli_server_process is not None and paroli_server_process.returncode is None:
        logger.warning("Paroli server process already seems to be running.")
        return

    if not settings.paroli_server.executable or not os.path.exists(settings.paroli_server.executable):
        logger.error(f"Paroli server executable not found or not set: {settings.paroli_server.executable}")
        logger.error("TTS will not be available.")
        return # Continue without TTS

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

        await asyncio.sleep(3.0) # Increased wait time

        if paroli_server_process.returncode is not None:
             logger.error(f"Paroli server process exited immediately after start with code: {paroli_server_process.returncode}")
             paroli_server_process = None
             if paroli_log_reader_task:
                paroli_log_reader_task.cancel()
                paroli_log_reader_task = None
        else:
             logger.info("Paroli server seems to be running.")

    except FileNotFoundError:
        logger.error(f"Paroli server executable not found at: {settings.paroli_server.executable}")
        paroli_server_process = None
    except Exception as e:
        logger.error(f"Failed to start paroli-server: {e}", exc_info=True)
        paroli_server_process = None

async def stop_paroli_server():
    """Останавливает процесс paroli-server."""
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
        logger.info(f"Stopping paroli-server process (PID: {paroli_server_process.pid})...")
        try:
            paroli_server_process.terminate()
            await asyncio.wait_for(paroli_server_process.wait(), timeout=5.0)
            logger.info("Paroli server process terminated gracefully.")
        except asyncio.TimeoutError:
            logger.warning("Paroli server process did not terminate gracefully, killing...")
            try:
                paroli_server_process.kill()
                await paroli_server_process.wait()
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

# --- Resource Initialization ---
def initialize_resources():
    global porcupine, cobra, vosk_model, vosk_recognizer
    logger.info("Initializing resources...")
    try:
        # Porcupine Wake Word
        porcupine = pvporcupine.create(
            access_key=settings.picovoice.access_key,
            keywords=["picovoice"], # Or your custom keyword file
            sensitivities=[0.5] # Adjust sensitivity as needed
        )
        logger.info(f"Porcupine initialized. Frame length: {porcupine.frame_length}, Sample rate: {porcupine.sample_rate}")
        if porcupine.sample_rate != SAMPLE_RATE:
            logger.warning(f"Porcupine sample rate ({porcupine.sample_rate}) != configured rate ({SAMPLE_RATE}).")
        # FRAME_LENGTH should match porcupine.frame_length ideally
        if porcupine.frame_length != FRAME_LENGTH:
             logger.warning(f"Porcupine frame length ({porcupine.frame_length}) != configured frame length ({FRAME_LENGTH}). Audio chunks might be misaligned.")
             # Consider setting FRAME_LENGTH = porcupine.frame_length if possible

        # Cobra VAD
        cobra = pvcobra.create(access_key=settings.picovoice.access_key)
        logger.info(f"Cobra VAD initialized. Frame length: {cobra.frame_length}, Sample rate: {cobra.sample_rate}")
        if cobra.sample_rate != SAMPLE_RATE:
             logger.warning(f"Cobra sample rate ({cobra.sample_rate}) != configured rate ({SAMPLE_RATE}).")
        if cobra.frame_length != FRAME_LENGTH:
             logger.warning(f"Cobra frame length ({cobra.frame_length}) != configured frame length ({FRAME_LENGTH}). VAD might be inaccurate.")

        # Vosk STT
        if not os.path.exists(settings.vosk.model_path):
            raise FileNotFoundError(f"Vosk model not found at {settings.vosk.model_path}")
        vosk_model = VoskModel(settings.vosk.model_path)
        # Use the global SAMPLE_RATE for Vosk
        vosk_recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
        vosk_recognizer.SetWords(True) # Enable word timings if needed later
        logger.info(f"Vosk model loaded from {settings.vosk.model_path}")

        # --- Paroli Sanity Check ---
        if not settings.paroli_server.executable:
             logger.warning("Path to paroli-server executable not set.")
        if not settings.paroli_server.ws_url:
             logger.warning("Paroli Server WebSocket URL is missing.")
        logger.info(f"Paroli Server Config: URL={settings.paroli_server.ws_url}, Format={settings.paroli_server.audio_format}")
        # --- End Paroli Check ---

        logger.info("All resources initialized successfully.")

    except pvporcupine.PorcupineError as e:
        logger.error(f"Porcupine initialization failed: {e}", exc_info=True)
        raise
    except pvcobra.CobraError as e:
        logger.error(f"Cobra initialization failed: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error initializing resources: {e}", exc_info=True)
        raise

# --- Audio Input Callback ---
def audio_input_callback(indata, frames, time, status):
    """
    This function is called by sounddevice for each new audio chunk.
    It puts the audio data into an asyncio queue for processing in the main async loop.
    """
    global current_state
    if status:
        logger.warning(f"Audio input status: {status}")
    
    # Only queue audio if we are waiting for wake word or actively listening
    if current_state in [AssistantState.WAITING_FOR_WAKEWORD, AssistantState.LISTENING]:
        try:
            # Use asyncio.run_coroutine_threadsafe for thread safety
            # Pass bytes directly
            asyncio.run_coroutine_threadsafe(audio_queue.put(indata.tobytes()), asyncio.get_event_loop())
        except Exception as e:
            logger.error(f"Error putting audio data into queue: {e}")

# --- Audio Output Thread ---
def audio_output_thread_func():
    """
    This function runs in a separate thread to play audio from the TTS queue.
    Using a separate thread avoids blocking the main asyncio loop with audio playback.
    """
    stream = None
    try:
        # Use default output device, configure samplerate based on Paroli output if known
        # Assuming Paroli outputs at SAMPLE_RATE for now, adjust if needed
        playback_sample_rate = settings.paroli_server.pcm_sample_rate if settings.paroli_server.audio_format == "pcm" else SAMPLE_RATE # Or a fixed known rate like 22050

        stream = sd.OutputStream(
            samplerate=playback_sample_rate,
            channels=CHANNELS,
            dtype=DTYPE, # Assuming Paroli outputs int16 PCM
            # device=OUTPUT_DEVICE_INDEX # Optional: specify output device index
        )
        stream.start()
        logger.info(f"Audio output stream started (Sample Rate: {playback_sample_rate}). Waiting for TTS data...")

        while True:
            chunk = tts_queue.get() # Blocks until an item is available
            if chunk is None: # Sentinel value to stop the thread
                logger.info("TTS end signal received. Stopping audio output.")
                break
            if isinstance(chunk, bytes) and len(chunk) > 0:
                try:
                    # Convert bytes to numpy array for sounddevice
                    # Ensure dtype matches stream dtype
                    numpy_chunk = np.frombuffer(chunk, dtype=np.int16)
                    stream.write(numpy_chunk)
                except Exception as e:
                    logger.error(f"Error playing TTS chunk: {e}")
            elif isinstance(chunk, bytes) and len(chunk) == 0:
                 logger.warning("Received empty TTS chunk, skipping playback.")
            else:
                 logger.warning(f"Received non-bytes or empty item in TTS queue: {type(chunk)}")

            tts_queue.task_done() # Notify queue that task is complete

    except sd.PortAudioError as pae:
        logger.error(f"PortAudio Error in output stream: {pae}")
        # Specific handling for common errors
        if "Invalid number of channels" in str(pae):
            logger.error("Check if your output device supports mono (1 channel) playback.")
        elif "Invalid sample rate" in str(pae):
            logger.error(f"Check if your output device supports the sample rate: {playback_sample_rate} Hz.")
        elif "Device unavailable" in str(pae):
            logger.error("Selected audio output device is unavailable. Check connection and OS settings.")
        else:
            logger.error("An unexpected PortAudio error occurred.")
    except Exception as e:
        logger.error(f"Error in audio output thread: {e}", exc_info=True)
    finally:
        if stream:
            try:
                # Wait for buffer to finish playing before closing
                # stream.stop() might not be sufficient, adding a small sleep
                # Check if stream has stop method and is active before calling
                if hasattr(stream, 'stopped') and not stream.stopped:
                    # Give some time for the buffer to clear, adjust as needed
                    # sd.sleep(int(stream.latency * 1000)) # Wait for estimated latency
                    stream.stop()
                stream.close()
                logger.info("Audio output stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error stopping/closing output stream: {e}")
        # Ensure the sentinel is processed if the loop exited unexpectedly
        if not tts_queue.empty():
            try:
                while tts_queue.get_nowait() is not None:
                    tts_queue.task_done()
            except queue.Empty:
                pass # Queue is now empty
        logger.info("Audio output thread finished.")


# --- TTS Synthesis (Directly calls Paroli) ---
async def synthesize_and_send_tts_local(text: str):
    """Synthesizes text using Paroli Server and puts audio chunks onto the tts_queue."""
    global current_state

    if not settings.paroli_server.ws_url:
        logger.error("Paroli WebSocket URL not configured. Cannot synthesize speech.")
        # Transition back to waiting state even if TTS fails
        current_state = AssistantState.WAITING_FOR_WAKEWORD
        logger.info("-> State: WAITING_FOR_WAKEWORD")
        return

    logger.info(f"Starting TTS for: '{text}'")
    current_state = AssistantState.SPEAKING
    logger.info("-> State: SPEAKING")

    try:
        async with websockets.connect(settings.paroli_server.ws_url, timeout=10.0) as paroli_ws: # Added connection timeout
            logger.info(f"Connected to Paroli Server for TTS: {settings.paroli_server.ws_url}")

            request_payload = {
                "text": text,
                "audio_format": settings.paroli_server.audio_format
            }
            if settings.paroli_server.speaker_id is not None: request_payload["speaker_id"] = settings.paroli_server.speaker_id
            if settings.paroli_server.length_scale is not None: request_payload["length_scale"] = settings.paroli_server.length_scale
            if settings.paroli_server.noise_scale is not None: request_payload["noise_scale"] = settings.paroli_server.noise_scale
            if settings.paroli_server.noise_w is not None: request_payload["noise_w"] = settings.paroli_server.noise_w

            await paroli_ws.send(json.dumps(request_payload))
            logger.debug("Sent TTS request to Paroli.")

            chunks_received = 0
            last_status_ok = False
            while True:
                try:
                    message = await asyncio.wait_for(paroli_ws.recv(), timeout=30.0) # Timeout for receiving messages
                except asyncio.TimeoutError:
                    logger.error("Timeout waiting for message from Paroli Server.")
                    break
                except websockets.exceptions.ConnectionClosedOK:
                    logger.info("Paroli Server closed connection cleanly.")
                    if not last_status_ok:
                         logger.warning("Paroli connection closed OK before receiving final 'ok' status.")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.error(f"Paroli Server connection closed unexpectedly: {e}")
                    break
                except Exception as recv_err:
                    logger.error(f"Error receiving message from Paroli Server: {recv_err}", exc_info=True)
                    break

                if isinstance(message, bytes):
                    if len(message) > 0:
                        tts_queue.put(message) # Put raw bytes onto the queue
                        chunks_received += 1
                    else:
                         logger.warning("Received empty audio chunk from Paroli.")
                elif isinstance(message, str):
                    logger.info(f"Received final status from Paroli: {message}")
                    try:
                        status_data = json.loads(message)
                        if status_data.get("status") == "ok":
                            last_status_ok = True
                        else:
                            logger.error(f"Paroli TTS failed: {status_data.get('message', 'Unknown error')}")
                    except json.JSONDecodeError:
                        logger.error(f"Could not decode final status JSON from Paroli: {message}")
                    break # End receiving loop
                else:
                    logger.warning(f"Received unexpected message type from Paroli: {type(message)}")

            if last_status_ok:
                 logger.info(f"Finished receiving TTS audio ({chunks_received} chunks).")
            else:
                 logger.error("TTS synthesis did not complete successfully according to Paroli status.")


    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid Paroli Server WebSocket URL: {settings.paroli_server.ws_url}")
    except ConnectionRefusedError:
         logger.error(f"Connection refused by Paroli Server at {settings.paroli_server.ws_url}. Is it running?")
    except asyncio.TimeoutError: # Connection timeout
         logger.error(f"Timeout connecting to Paroli Server at {settings.paroli_server.ws_url}.")
    except Exception as e:
        logger.error(f"Paroli Server TTS Error: {e}", exc_info=True)
    finally:
        # Signal the end of TTS audio to the output thread
        tts_queue.put(None)
        # Always transition back to waiting for wake word after attempting TTS
        current_state = AssistantState.WAITING_FOR_WAKEWORD
        logger.info("-> State: WAITING_FOR_WAKEWORD (after TTS)")


# --- Main Processing Loop ---
async def process_audio_loop():
    global current_state, silence_frames_count, listening_frames_count, vosk_recognizer

    logger.info("Starting main audio processing loop...")
    logger.info("-> State: WAITING_FOR_WAKEWORD")

    while True:
        try:
            # Get audio chunk from the input callback queue
            audio_chunk = await audio_queue.get()

            # Ensure chunk size matches expected frame length (important!)
            expected_bytes = FRAME_LENGTH * (np.dtype(DTYPE).itemsize) # bytes per frame
            if len(audio_chunk) != expected_bytes:
                # This might happen if sounddevice settings don't align perfectly
                # or due to timing issues. It can cause errors in engines.
                logger.warning(f"Received audio chunk size ({len(audio_chunk)} bytes) "
                               f"does not match expected ({expected_bytes} bytes). Skipping frame.")
                # Simple strategy: skip this frame. More complex handling (buffering, resampling) might be needed.
                audio_queue.task_done()
                continue

            try:
                 # Convert bytes to list of int16 samples for Picovoice/Cobra
                pcm = struct.unpack_from(f"{FRAME_LENGTH}h", audio_chunk)
            except struct.error as e:
                logger.error(f"Error unpacking audio chunk: {e}. Chunk length: {len(audio_chunk)}, Expected shorts: {FRAME_LENGTH}")
                audio_queue.task_done()
                continue

            # --- State Machine Logic ---
            if current_state == AssistantState.WAITING_FOR_WAKEWORD:
                try:
                    # Check frame length compatibility with Porcupine
                    if len(pcm) == porcupine.frame_length:
                        keyword_index = porcupine.process(pcm)
                        if keyword_index >= 0:
                            logger.info("Wake word detected!")
                            current_state = AssistantState.LISTENING
                            logger.info("-> State: LISTENING")
                            silence_frames_count = 0
                            listening_frames_count = 0
                            vosk_recognizer.Reset() # Reset Vosk for new utterance
                            # Optional: Play a sound indication
                            # tts_queue.put(load_wav_file("path/to/start_sound.wav"))
                    else:
                         # This should ideally not happen if FRAME_LENGTH is set correctly
                         logger.warning(f"PCM length {len(pcm)} != Porcupine frame length {porcupine.frame_length}. Skipping WWD check.")

                except pvporcupine.PorcupineError as e:
                    logger.error(f"Porcupine processing error: {e}")
                    # Optionally reset state or handle error

            elif current_state == AssistantState.LISTENING:
                listening_frames_count += 1
                is_voiced = False

                # VAD Check (Cobra)
                try:
                     # Check frame length compatibility with Cobra
                    if len(pcm) == cobra.frame_length:
                        voice_probability = cobra.process(pcm)
                        is_voiced = voice_probability > settings.vad.probability_threshold
                    else:
                        logger.warning(f"PCM length {len(pcm)} != Cobra frame length {cobra.frame_length}. Assuming silence for VAD.")
                        is_voiced = False # Treat as silence if frame size is wrong
                except pvcobra.CobraError as e:
                    logger.error(f"Cobra processing error: {e}")
                    is_voiced = False # Treat as silence on error

                # STT Processing (Vosk)
                vosk_accepted_waveform = vosk_recognizer.AcceptWaveform(audio_chunk)
                partial_result_json = vosk_recognizer.PartialResult()
                partial_result = json.loads(partial_result_json)
                # logger.debug(f"Partial: {partial_result.get('partial', '')}") # Optional debug

                # VAD Logic for End of Speech
                if not is_voiced:
                    silence_frames_count += 1
                else:
                    silence_frames_count = 0 # Reset counter if voice detected

                # Check conditions for ending listening phase
                min_listen_frames = settings.vad.min_listening_frames
                max_silence_frames = settings.vad.silence_frames

                grace_period_over = listening_frames_count >= min_listen_frames
                silence_threshold_met = silence_frames_count >= max_silence_frames

                if grace_period_over and silence_threshold_met:
                    logger.info(f"VAD detected end of speech (Silence frames: {silence_frames_count}). Finalizing...")
                    final_result_json = vosk_recognizer.FinalResult()
                    final_result_data = json.loads(final_result_json)
                    final_transcript = final_result_data.get("text", "").strip()
                    logger.debug(f"Vosk FinalResult JSON: {final_result_json}")

                    if final_transcript:
                        logger.info(f"Final Transcript: '{final_transcript}'")
                        current_state = AssistantState.PROCESSING_LLM
                        logger.info("-> State: PROCESSING_LLM")

                        # --- Call LLM (Potentially Blocking - Run in Executor) ---
                        try:
                            loop = asyncio.get_running_loop()
                            # Assuming ask_lumi is synchronous
                            lumi_response = await loop.run_in_executor(
                                None, # Use default thread pool executor
                                ask_lumi,
                                final_transcript,
                                # thread_id="local_device" # Pass a consistent ID if needed
                            )
                            logger.info(f"Lumi Response: '{lumi_response}'")

                            if lumi_response:
                                # --- Start TTS ---
                                # Don't await this directly, let it run in background
                                asyncio.create_task(synthesize_and_send_tts_local(lumi_response))
                            else:
                                logger.warning("LLM returned empty response.")
                                # No TTS needed, go back to waiting
                                current_state = AssistantState.WAITING_FOR_WAKEWORD
                                logger.info("-> State: WAITING_FOR_WAKEWORD (empty LLM response)")

                        except Exception as llm_err:
                            logger.error(f"Error during LLM call: {llm_err}", exc_info=True)
                            # Optional: Synthesize an error message
                            asyncio.create_task(synthesize_and_send_tts_local("Sorry, I encountered an error processing your request."))
                            # State will be set back to WAITING in synthesize_and_send_tts_local's finally block

                    else:
                        logger.info("No final transcript obtained. Returning to wake word listening.")
                        current_state = AssistantState.WAITING_FOR_WAKEWORD
                        logger.info("-> State: WAITING_FOR_WAKEWORD (no transcript)")
                        # Optional: Play a cancel sound
                        # tts_queue.put(load_wav_file("path/to/cancel_sound.wav"))

            # elif current_state == AssistantState.PROCESSING_LLM:
                # Input audio is ignored while processing LLM
                # pass

            # elif current_state == AssistantState.SPEAKING:
                # Input audio is ignored while speaking
                # pass

            # Mark the task as done in the input queue
            audio_queue.task_done()

        except asyncio.CancelledError:
             logger.info("Audio processing loop cancelled.")
             break
        except Exception as e:
            logger.error(f"Error in main audio processing loop: {e}", exc_info=True)
            # Attempt to recover state if possible
            if current_state != AssistantState.WAITING_FOR_WAKEWORD:
                 logger.warning("Resetting state to WAITING_FOR_WAKEWORD due to error.")
                 current_state = AssistantState.WAITING_FOR_WAKEWORD
                 # Clear queues maybe?
            # Avoid busy-looping on continuous errors
            await asyncio.sleep(0.5)


# --- Main Execution ---
async def main():
    global audio_input_stream # Make stream accessible for stopping

    # Check audio devices
    logger.info("Available audio devices:")
    logger.info(sd.query_devices())
    # You might need to set input/output device indices in settings.py
    input_device_index = getattr(settings.audio, 'input_device_index', None)
    output_device_index = getattr(settings.audio, 'output_device_index', None)
    logger.info(f"Using Input Device Index: {input_device_index if input_device_index is not None else 'Default'}")
    logger.info(f"Using Output Device Index: {output_device_index if output_device_index is not None else 'Default'}")


    initialize_resources() # Load models, etc.
    await start_paroli_server() # Start TTS server process

    # Start audio output thread
    import threading
    output_thread = threading.Thread(target=audio_output_thread_func, daemon=True)
    output_thread.start()

    # Start audio input stream
    audio_input_stream = None
    try:
        logger.info(f"Attempting to start audio input stream (Rate: {SAMPLE_RATE}, Frame: {FRAME_LENGTH}, Dtype: {DTYPE})")
        audio_input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_LENGTH, # Process audio in chunks matching engine frame lengths
            channels=CHANNELS,
            dtype=DTYPE,
            device=input_device_index, # Use specified or default device
            callback=audio_input_callback
        )
        audio_input_stream.start()
        logger.info("Audio input stream started.")

        # Start the main processing loop
        processing_task = asyncio.create_task(process_audio_loop())

        # Keep running until interrupted
        await processing_task # Wait for the processing loop to finish (e.g., on cancellation)

    except sd.PortAudioError as pae:
        logger.error(f"PortAudio Error starting input stream: {pae}")
        if "Invalid number of channels" in str(pae):
            logger.error("Check if your input device supports mono (1 channel) recording.")
        elif "Invalid sample rate" in str(pae):
            logger.error(f"Check if your input device supports the sample rate: {SAMPLE_RATE} Hz.")
        elif "Device unavailable" in str(pae):
            logger.error("Selected audio input device is unavailable. Check connection and OS settings.")
        else:
            logger.error("An unexpected PortAudio error occurred.")
        logger.error("Application cannot continue without audio input.")

    except Exception as e:
        logger.error(f"Error during main execution: {e}", exc_info=True)
    finally:
        logger.info("Shutting down...")
        # Stop audio input stream
        if audio_input_stream and not audio_input_stream.stopped:
            logger.info("Stopping audio input stream...")
            audio_input_stream.stop()
            audio_input_stream.close()
            logger.info("Audio input stream stopped and closed.")

        # Signal output thread to stop and wait for it
        if output_thread.is_alive():
             logger.info("Signaling audio output thread to stop...")
             tts_queue.put(None) # Send sentinel
             # Wait briefly for thread to process sentinel and exit
             output_thread.join(timeout=5.0)
             if output_thread.is_alive():
                 logger.warning("Audio output thread did not stop gracefully.")

        # Stop Paroli server
        await stop_paroli_server()

        # Cleanup Picovoice resources
        if porcupine:
            porcupine.delete()
            logger.info("Porcupine resources released.")
        if cobra:
            cobra.delete()
            logger.info("Cobra resources released.")

        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting.")
    except Exception as e:
         logger.critical(f"Unhandled exception in main execution block: {e}", exc_info=True)
    finally:
         # Ensure logging finishes
         logging.shutdown()