import asyncio
import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from settings import settings # Import shared settings

logger = logging.getLogger(__name__)

class LocalAudioManager:
    """Manages local audio input (microphone) and output (speakers)."""

    def __init__(self, manager_process_audio_cb, loop):
        """
        Initializes the LocalAudioManager.

        Args:
            manager_process_audio_cb: Async callback function from ConnectionManager
                                      to process audio chunks.
            loop: The main asyncio event loop.
        """
        if not settings.local_audio.enabled:
            logger.info("Local audio is disabled in settings.")
            self._enabled = False
            return

        self._enabled = True
        self._manager_process_audio_cb = manager_process_audio_cb
        self._loop = loop

        self._input_stream = None
        self._output_stream = None
        self._input_thread = None
        self._output_thread = None
        self._output_queue = queue.Queue() # Queue for TTS audio data bytes
        self._stop_event = threading.Event()
        self._is_input_running = False
        self._is_output_running = False
        self._is_paused = False # Flag to pause input when WebSocket is active

        # Validate devices
        try:
            sd.check_input_settings(device=settings.local_audio.input_device_index,
                                    samplerate=settings.local_audio.sample_rate,
                                    channels=settings.local_audio.channels)
            sd.check_output_settings(device=settings.local_audio.output_device_index,
                                     samplerate=settings.local_audio.tts_output_sample_rate, # Use TTS rate for output
                                     channels=settings.local_audio.channels) # Assuming mono TTS output
            logger.info("Local audio devices validated.")
            logger.info(f"Input: Device={settings.local_audio.input_device_index}, Rate={settings.local_audio.sample_rate}, Block={settings.local_audio.frame_length}")
            logger.info(f"Output: Device={settings.local_audio.output_device_index}, Rate={settings.local_audio.tts_output_sample_rate}")
        except Exception as e:
            logger.error(f"Local audio device configuration error: {e}", exc_info=True)
            logger.error("Disabling local audio due to configuration error.")
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _input_callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        """Callback function for the sounddevice input stream."""
        if status:
            logger.warning(f"Sounddevice input status: {status}")
        if self._is_paused: # Don't process if paused (WebSocket active)
            return
        if not self._manager_process_audio_cb:
            return

        # Ensure data is int16 bytes, matching WebSocket format
        # Convert float32 numpy array to int16 bytes
        try:
            # Ensure the input data is float32, scale to int16 range, convert type
            if indata.dtype != np.float32:
                # This shouldn't happen with default sounddevice streams, but good practice
                 indata = indata.astype(np.float32)

            # Scale and clip float32 data [-1.0, 1.0] to int16 [-32768, 32767]
            scaled_data = indata * 32767.0
            clipped_data = np.clip(scaled_data, -32768, 32767)
            int16_data = clipped_data.astype(np.int16)

            # Convert numpy array to bytes
            audio_bytes = int16_data.tobytes()

            # Schedule the async processing function in the main event loop
            # Important: Use run_coroutine_threadsafe for thread safety
            asyncio.run_coroutine_threadsafe(
                self._manager_process_audio_cb(audio_bytes), # Pass bytes
                self._loop
            )
        except Exception as e:
            logger.error(f"Error processing audio in input callback: {e}", exc_info=True)


    def _output_worker(self):
        """Worker thread for playing audio from the queue."""
        stream = None
        try:
            stream = sd.OutputStream(
                samplerate=settings.local_audio.tts_output_sample_rate,
                blocksize=1024, # Adjust block size if needed for latency/performance
                device=settings.local_audio.output_device_index,
                channels=settings.local_audio.channels, # Assuming mono output
                dtype='int16' # Assuming TTS provides int16 PCM
            )
            stream.start()
            logger.info("Local audio output stream started.")
            self._is_output_running = True
            while not self._stop_event.is_set():
                try:
                    # Wait for audio data with a timeout
                    audio_bytes = self._output_queue.get(timeout=0.5)
                    if audio_bytes is None: # Sentinel value to stop
                         break
                    # Convert bytes back to numpy array for sounddevice
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    stream.write(audio_array)
                    self._output_queue.task_done()
                except queue.Empty:
                    continue # No data, check stop event again
                except Exception as e:
                    logger.error(f"Error writing to output stream: {e}", exc_info=True)
                    # Avoid busy-looping on error
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to start or run local audio output stream: {e}", exc_info=True)
        finally:
            if stream:
                try:
                    if not stream.stopped:
                        stream.stop()
                    stream.close()
                    logger.info("Local audio output stream stopped and closed.")
                except Exception as e:
                    logger.error(f"Error closing output stream: {e}", exc_info=True)
            self._is_output_running = False
            # Ensure any waiting get() calls unblock if thread exits unexpectedly
            self._output_queue.put(None)


    def start_input(self):
        """Starts the audio input stream."""
        if not self._enabled or self._is_input_running:
            return
        logger.info("Starting local audio input stream...")
        try:
            self._is_paused = False # Ensure not paused on start
            self._input_stream = sd.InputStream(
                samplerate=settings.local_audio.sample_rate,
                blocksize=settings.local_audio.frame_length, # Match processing chunk size
                device=settings.local_audio.input_device_index,
                channels=settings.local_audio.channels,
                dtype='float32', # Let sounddevice handle native format, convert in callback
                callback=self._input_callback
            )
            self._input_stream.start()
            self._is_input_running = True
            logger.info("Local audio input stream started.")
        except Exception as e:
            logger.error(f"Failed to start local audio input stream: {e}", exc_info=True)
            self._input_stream = None
            self._is_input_running = False


    def stop_input(self):
        """Stops the audio input stream."""
        if not self._enabled or not self._is_input_running:
            return
        logger.info("Stopping local audio input stream...")
        if self._input_stream:
            try:
                self._input_stream.stop()
                self._input_stream.close()
                logger.info("Local audio input stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error stopping input stream: {e}", exc_info=True)
            finally:
                self._input_stream = None
        self._is_input_running = False
        self._is_paused = False # Reset pause state


    def pause_input(self):
        """Pauses processing of input audio (e.g., when WebSocket connects)."""
        if self._is_input_running and not self._is_paused:
            self._is_paused = True
            logger.info("Local audio input processing paused.")

    def resume_input(self):
        """Resumes processing of input audio (e.g., when WebSocket disconnects)."""
        if self._is_input_running and self._is_paused:
            self._is_paused = False
            logger.info("Local audio input processing resumed.")


    def start_output(self):
        """Starts the audio output worker thread."""
        if not self._enabled or self._is_output_running:
            return
        logger.info("Starting local audio output worker...")
        self._stop_event.clear()
        # Clear queue from potential stale data or sentinel
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break
        self._output_thread = threading.Thread(target=self._output_worker, daemon=True)
        self._output_thread.start()


    def stop_output(self):
        """Stops the audio output worker thread."""
        if not self._enabled or not self._is_output_running:
            return
        logger.info("Stopping local audio output worker...")
        self._stop_event.set()
        self._output_queue.put(None) # Send sentinel to unblock worker
        if self._output_thread and self._output_thread.is_alive():
             self._output_thread.join(timeout=2.0) # Wait for thread to finish
             if self._output_thread.is_alive():
                 logger.warning("Output worker thread did not terminate gracefully.")
        self._output_thread = None
        self._is_output_running = False


    def play_tts_bytes(self, audio_bytes: bytes):
        """Adds TTS audio bytes to the output queue for playback."""
        if not self._enabled or not self._is_output_running or self._is_paused: # Also don't play if input is paused (WS active)
            if self._is_paused:
                logger.debug("Skipping local TTS playback because input is paused (WebSocket likely active).")
            elif not self._is_output_running:
                 logger.warning("Cannot play TTS locally: output worker not running.")
            return
        if audio_bytes:
            self._output_queue.put(audio_bytes)


    def shutdown(self):
        """Shuts down both input and output streams/workers."""
        logger.info("Shutting down LocalAudioManager...")
        self.stop_input()
        self.stop_output()
        logger.info("LocalAudioManager shutdown complete.")