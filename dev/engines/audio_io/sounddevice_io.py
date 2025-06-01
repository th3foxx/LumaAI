import asyncio
import logging
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from typing import Callable, Coroutine, Any, Dict, Optional

from .input_base import AudioInputEngineBase, ProcessAudioCallback
from .output_base import AudioOutputEngineBase
from settings import SoundDeviceSettings, AudioSettings 

logger = logging.getLogger(__name__)

class SoundDeviceInputEngine(AudioInputEngineBase):
    def __init__(self, process_audio_callback: Optional[ProcessAudioCallback], loop: Any, 
                 config: Dict[str, Any]): 
        self._process_audio_cb = process_audio_callback
        self._loop = loop
        self.sd_settings: SoundDeviceSettings = config["sounddevice_settings"]
        self.audio_settings: AudioSettings = config["audio_settings"]

        self._input_stream: Optional[sd.InputStream] = None
        self._is_running: bool = False
        self._is_paused: bool = False
        self._enabled: bool = self.sd_settings.enabled

        if self._enabled:
            if not self._process_audio_cb:
                logger.warning("SoundDeviceInput: process_audio_callback is None. Engine will be disabled.")
                self._enabled = False
            else:
                try:
                    sd.check_input_settings(
                        device=self.sd_settings.input_device_index,
                        samplerate=self.audio_settings.sample_rate,
                        channels=self.audio_settings.channels,
                        dtype='float32' # We request float32 for consistent processing
                    )
                    logger.info(f"SoundDeviceInput: Configured for Device={self.sd_settings.input_device_index or 'Default'}, "
                                f"Rate={self.audio_settings.sample_rate}Hz, Channels={self.audio_settings.channels}, "
                                f"Blocksize={self.audio_settings.frame_length} samples.")
                except Exception as e:
                    logger.error(f"SoundDeviceInput: Invalid input audio settings. Disabling engine. Error: {e}", exc_info=True)
                    self._enabled = False
        else:
            logger.info("SoundDeviceInput is disabled by settings.")

    @property
    def is_enabled(self) -> bool: return self._enabled
    @property
    def is_running(self) -> bool: return self._is_running
    @property
    def sample_rate(self) -> int: return self.audio_settings.sample_rate
    @property
    def frame_length(self) -> int: return self.audio_settings.frame_length
    @property
    def channels(self) -> int: return self.audio_settings.channels

    def _sd_input_callback(self, indata: np.ndarray, frames: int, time_info, status: sd.CallbackFlags):
        if not self._is_running or self._is_paused or not self._process_audio_cb:
            return
        if status: 
            logger.warning(f"Sounddevice input status: {status}")
            if status.input_overflow:
                logger.error("Sounddevice input overflow detected! Audio data may have been lost.")
            if status.input_underflow: # Should not happen for input
                logger.error("Sounddevice input underflow detected!")


        try:
            # Ensure data is float32, then convert to int16 bytes
            # Sounddevice should provide float32 if dtype='float32' was requested for InputStream
            if indata.dtype != np.float32:
                # This case should ideally not happen if stream is configured for float32
                logger.warning(f"SoundDeviceInput: Received data with dtype {indata.dtype}, expected float32. Attempting conversion.")
                indata = indata.astype(np.float32)
            
            # Scale float32 from [-1.0, 1.0] to int16 [-32768, 32767]
            scaled_data = indata * 32767.0 
            clipped_data = np.clip(scaled_data, -32768.0, 32767.0) # Clip to ensure valid int16 range
            audio_bytes = clipped_data.astype(np.int16).tobytes()
            
            asyncio.run_coroutine_threadsafe(self._process_audio_cb(audio_bytes), self._loop)
        except Exception as e:
            logger.error(f"Error in SoundDeviceInput callback: {e}", exc_info=True)

    def start(self):
        if not self._enabled:
            logger.debug("SoundDeviceInput: Cannot start, engine is disabled.")
            return
        if self._is_running:
            logger.debug("SoundDeviceInput: Cannot start, engine is already running.")
            return
        if not self._process_audio_cb: # Should be caught by __init__ but as a safeguard
            logger.error("SoundDeviceInput: process_audio_callback is not set. Cannot start.")
            self._enabled = False
            return

        logger.info("SoundDeviceInput: Starting audio stream...")
        try:
            self._is_paused = False
            self._input_stream = sd.InputStream(
                samplerate=self.audio_settings.sample_rate,
                blocksize=self.audio_settings.frame_length,
                device=self.sd_settings.input_device_index,
                channels=self.audio_settings.channels,
                dtype='float32', # Request float32
                callback=self._sd_input_callback
            )
            self._input_stream.start()
            self._is_running = True
            logger.info("SoundDeviceInput: Audio stream started successfully.")
        except Exception as e:
            logger.error(f"SoundDeviceInput: Failed to start audio stream: {e}", exc_info=True)
            if self._input_stream:
                try:
                    self._input_stream.close()
                except Exception as e_close:
                    logger.error(f"SoundDeviceInput: Error closing stream after failed start: {e_close}")
            self._input_stream = None
            self._is_running = False

    def stop(self):
        if not self._is_running:
            # logger.debug("SoundDeviceInput: Cannot stop, engine is not running.")
            return
        
        logger.info("SoundDeviceInput: Stopping audio stream...")
        self._is_running = False # Set this first to prevent callback processing
        self._is_paused = False # Reset pause state

        if self._input_stream:
            stream_to_close = self._input_stream
            self._input_stream = None # Nullify before closing
            try:
                if not stream_to_close.closed:
                    stream_to_close.stop()
                    stream_to_close.close()
                logger.info("SoundDeviceInput: Audio stream stopped and closed.")
            except Exception as e: 
                logger.error(f"SoundDeviceInput: Error stopping or closing audio stream: {e}", exc_info=True)
        else:
            logger.info("SoundDeviceInput: Stream was already None when stop was called.")


    def pause(self):
        if self._is_running and not self._is_paused:
            self._is_paused = True
            logger.info("SoundDeviceInput: Paused.")
        elif not self._is_running:
            logger.debug("SoundDeviceInput: Cannot pause, not running.")
        elif self._is_paused:
            logger.debug("SoundDeviceInput: Already paused.")


    def resume(self):
        if self._is_running and self._is_paused:
            self._is_paused = False
            logger.info("SoundDeviceInput: Resumed.")
        elif not self._is_running:
            logger.debug("SoundDeviceInput: Cannot resume, not running.")
        elif not self._is_paused:
            logger.debug("SoundDeviceInput: Not paused, no need to resume.")
            
    async def shutdown(self): 
        logger.info("SoundDeviceInput: Shutting down...")
        self.stop()
        logger.info("SoundDeviceInput: Shutdown complete.")


class SoundDeviceOutputEngine(AudioOutputEngineBase):
    def __init__(self, config: Dict[str, Any]): 
        self.sd_settings: SoundDeviceSettings = config["sounddevice_settings"]
        # Global audio settings might specify default channels, but TTS output is typically mono.
        # We'll use 1 channel for output unless TTS provides multi-channel (rare).
        self.output_channels: int = 1 # Default to mono for TTS
        
        self._output_queue: queue.Queue[Optional[Tuple[bytes, int]]] = queue.Queue() # Stores (audio_bytes, sample_rate) or None
        self._output_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._is_running: bool = False # Indicates if the worker thread is active and processing
        self._enabled: bool = self.sd_settings.enabled

        self.current_stream: Optional[sd.OutputStream] = None
        self.current_stream_sample_rate: Optional[int] = None

        if self._enabled:
            try:
                # Initial check with default/configured rate and mono channel
                sd.check_output_settings(
                    device=self.sd_settings.output_device_index,
                    samplerate=self.sd_settings.tts_output_sample_rate, 
                    channels=self.output_channels,
                    dtype='int16' # Assuming TTS provides int16 PCM
                )
                logger.info(f"SoundDeviceOutput: Configured for Device={self.sd_settings.output_device_index or 'Default'}, "
                            f"Default Rate={self.sd_settings.tts_output_sample_rate}Hz, Channels={self.output_channels}.")
            except Exception as e:
                logger.error(f"SoundDeviceOutput: Invalid output audio settings. Disabling engine. Error: {e}", exc_info=True)
                self._enabled = False
        else:
            logger.info("SoundDeviceOutput is disabled by settings.")
            
    @property
    def is_enabled(self) -> bool: return self._enabled
    @property
    def is_running(self) -> bool: return self._is_running # Worker thread is active

    def _output_worker(self):
        logger.info("SoundDeviceOutput: Worker thread started.")
        self._is_running = True # Mark worker as active
        
        while not self._stop_event.is_set():
            try:
                item = self._output_queue.get(timeout=0.2) # Shorter timeout for responsiveness
                if item is None: # Sentinel to stop
                    logger.debug("SoundDeviceOutput: Worker received stop sentinel.")
                    break 
                
                audio_bytes, sample_rate_for_chunk = item

                # Ensure stream is open and matches sample rate
                if self.current_stream is None or sample_rate_for_chunk != self.current_stream_sample_rate:
                    if self.current_stream:
                        logger.info(f"SoundDeviceOutput: Changing sample rate from {self.current_stream_sample_rate}Hz to {sample_rate_for_chunk}Hz.")
                        try:
                            if not self.current_stream.closed:
                                self.current_stream.stop()
                                self.current_stream.close()
                        except Exception as e:
                            logger.error(f"SoundDeviceOutput: Error closing existing stream: {e}")
                        self.current_stream = None
                    
                    try:
                        logger.info(f"SoundDeviceOutput: Opening new stream with SR={sample_rate_for_chunk}Hz, Channels={self.output_channels}.")
                        self.current_stream = sd.OutputStream(
                            samplerate=sample_rate_for_chunk,
                            blocksize=0, # Let sounddevice choose blocksize for OutputStream
                            device=self.sd_settings.output_device_index,
                            channels=self.output_channels, # Use configured channels (typically 1 for TTS)
                            dtype='int16' 
                        )
                        self.current_stream.start()
                        self.current_stream_sample_rate = sample_rate_for_chunk
                    except Exception as e:
                        logger.error(f"SoundDeviceOutput: Failed to open stream with SR={sample_rate_for_chunk}Hz: {e}", exc_info=True)
                        self.current_stream = None
                        self.current_stream_sample_rate = None
                        self._output_queue.task_done() 
                        continue # Skip this chunk

                # Play audio if stream is ready
                if self.current_stream and self.current_stream_sample_rate == sample_rate_for_chunk:
                    try:
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        self.current_stream.write(audio_array)
                    except Exception as e_write:
                        logger.error(f"SoundDeviceOutput: Error writing to audio stream: {e_write}", exc_info=True)
                        # Consider closing and reopening stream on write errors
                        try:
                            if not self.current_stream.closed:
                                self.current_stream.stop()
                                self.current_stream.close()
                        except Exception: pass # Ignore errors during error handling
                        self.current_stream = None
                        self.current_stream_sample_rate = None
                else:
                    logger.warning(f"SoundDeviceOutput: Stream not ready or sample rate mismatch for chunk. SR_expected={sample_rate_for_chunk}, SR_stream={self.current_stream_sample_rate}")
                
                self._output_queue.task_done()

            except queue.Empty:
                # If queue is empty and a stream is open but no new data for a while,
                # we might consider closing the stream to free resources, but this adds complexity.
                # For now, keep stream open if it was used.
                time.sleep(0.01) # Small sleep to prevent busy loop on timeout
                continue
            except Exception as e:
                logger.error(f"Error in SoundDeviceOutput worker loop: {e}", exc_info=True)
                time.sleep(0.1) # Avoid rapid error logging
        
        # Cleanup stream when worker stops
        if self.current_stream:
            logger.info("SoundDeviceOutput worker: Stopping and closing final audio stream.")
            try:
                if not self.current_stream.closed:
                    self.current_stream.stop() # Ensure it's stopped before closing
                    self.current_stream.close()
            except Exception as e:
                logger.error(f"SoundDeviceOutput: Error closing stream on worker exit: {e}")
            self.current_stream = None
            self.current_stream_sample_rate = None
        
        self._is_running = False # Mark worker as stopped
        logger.info("SoundDeviceOutput: Worker thread stopped.")


    def start(self):
        if not self._enabled:
            logger.debug("SoundDeviceOutput: Cannot start, engine is disabled.")
            return
        if self._is_running or (self._output_thread and self._output_thread.is_alive()):
            logger.debug("SoundDeviceOutput: Cannot start, worker thread is already running or active.")
            return

        logger.info("SoundDeviceOutput: Attempting to start worker thread...")
        self._stop_event.clear()
        # Clear any stale items from the queue before starting
        while not self._output_queue.empty():
            try: self._output_queue.get_nowait()
            except queue.Empty: break
        
        self._output_thread = threading.Thread(target=self._output_worker, daemon=True, name="SoundDeviceOutputThread")
        self._output_thread.start()
        
        # Give the thread a moment to set _is_running
        time.sleep(0.05) 
        if not self._is_running:
            logger.warning("SoundDeviceOutput: Worker thread may not have started correctly (is_running is False).")
        else:
            logger.info("SoundDeviceOutput: Worker thread started successfully.")


    def stop(self):
        if not self._enabled: # No need to stop if it was never enabled to start
            return
        
        if not self._is_running and not (self._output_thread and self._output_thread.is_alive()):
            # logger.debug("SoundDeviceOutput: Stop called, but worker not running or thread inactive.")
            return

        logger.info("SoundDeviceOutput: Attempting to stop worker thread...")
        self._stop_event.set()
        
        # Clear the queue and add sentinel to ensure worker exits if blocked on get()
        while not self._output_queue.empty():
            try: self._output_queue.get_nowait()
            except queue.Empty: break
        self._output_queue.put(None) 
        
        if self._output_thread and self._output_thread.is_alive():
             logger.debug("SoundDeviceOutput: Joining worker thread...")
             self._output_thread.join(timeout=1.0) # Shorter timeout, should exit quickly
             if self._output_thread.is_alive():
                 logger.warning("SoundDeviceOutput: Worker thread did not terminate gracefully after 1s.")
        
        self._output_thread = None
        # _is_running should be set to False by the worker thread itself upon exit.
        # If join times out or thread had an issue, force it.
        if self._is_running: 
            logger.warning("SoundDeviceOutput: Worker thread still marked as running after stop attempt. Forcing status.")
            self._is_running = False 

        logger.info("SoundDeviceOutput: Stop sequence complete.")


    def play_tts_bytes(self, audio_bytes: bytes, sample_rate: Optional[int] = None):
        if not self._enabled:
            # logger.debug("SoundDeviceOutput is disabled, cannot play TTS.")
            return
        
        if not self._is_running: 
            logger.warning("SoundDeviceOutput: Worker not running. Attempting to start it to play TTS.")
            self.start() 
            if not self._is_running: 
                 logger.error("SoundDeviceOutput: Failed to start worker, cannot play TTS.")
                 return

        if not audio_bytes:
            logger.debug("SoundDeviceOutput: play_tts_bytes called with empty audio_bytes. Skipping.")
            return

        sr_to_use = sample_rate if sample_rate is not None else self.sd_settings.tts_output_sample_rate
        # logger.debug(f"SoundDeviceOutput: Queuing {len(audio_bytes)} bytes with SR={sr_to_use}Hz.")
        try:
            self._output_queue.put((audio_bytes, sr_to_use), block=False) # Non-blocking put
        except queue.Full:
            logger.warning("SoundDeviceOutput: Output queue is full. TTS audio may be dropped or delayed.")


    async def shutdown(self): 
        logger.info("SoundDeviceOutput: Shutting down...")
        self.stop()
        # Ensure queue is empty after stop, though stop() should handle it.
        while not self._output_queue.empty():
            try: self._output_queue.get_nowait()
            except queue.Empty: break
        logger.info("SoundDeviceOutput: Shutdown complete.")