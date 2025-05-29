import asyncio
import logging
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from typing import Callable, Coroutine, Any, Dict, Optional # Added Optional

from .input_base import AudioInputEngineBase, ProcessAudioCallback
from .output_base import AudioOutputEngineBase
from settings import SoundDeviceSettings, AudioSettings 

logger = logging.getLogger(__name__)

class SoundDeviceInputEngine(AudioInputEngineBase):
    def __init__(self, process_audio_callback: ProcessAudioCallback, loop: Any, 
                 config: Dict[str, Any]): 
        self._process_audio_cb = process_audio_callback
        self._loop = loop
        self.sd_settings: SoundDeviceSettings = config["sounddevice_settings"]
        self.audio_settings: AudioSettings = config["audio_settings"]

        self._input_stream = None
        self._is_running = False
        self._is_paused = False
        self._enabled = self.sd_settings.enabled

        if self._enabled:
            try:
                sd.check_input_settings(
                    device=self.sd_settings.input_device_index,
                    samplerate=self.audio_settings.sample_rate,
                    channels=self.audio_settings.channels
                )
                logger.info(f"SoundDeviceInput: Device={self.sd_settings.input_device_index}, "
                            f"Rate={self.audio_settings.sample_rate}, Block={self.audio_settings.frame_length}")
            except Exception as e:
                logger.error(f"SoundDeviceInput configuration error: {e}", exc_info=True)
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
        if status: logger.warning(f"Sounddevice input status: {status}")
        if self._is_paused or not self._process_audio_cb: return

        try:
            if indata.dtype != np.float32: indata = indata.astype(np.float32)
            scaled_data = indata * 32767.0
            clipped_data = np.clip(scaled_data, -32768, 32767)
            audio_bytes = clipped_data.astype(np.int16).tobytes()
            asyncio.run_coroutine_threadsafe(self._process_audio_cb(audio_bytes), self._loop)
        except Exception as e:
            logger.error(f"Error in SoundDeviceInput callback: {e}", exc_info=True)

    def start(self):
        if not self._enabled or self._is_running: return
        logger.info("Starting SoundDeviceInput stream...")
        try:
            self._is_paused = False
            self._input_stream = sd.InputStream(
                samplerate=self.audio_settings.sample_rate,
                blocksize=self.audio_settings.frame_length,
                device=self.sd_settings.input_device_index,
                channels=self.audio_settings.channels,
                dtype='float32',
                callback=self._sd_input_callback
            )
            self._input_stream.start()
            self._is_running = True
            logger.info("SoundDeviceInput stream started.")
        except Exception as e:
            logger.error(f"Failed to start SoundDeviceInput stream: {e}", exc_info=True)
            self._input_stream = None
            self._is_running = False

    def stop(self):
        if not self._enabled or not self._is_running: return
        logger.info("Stopping SoundDeviceInput stream...")
        if self._input_stream:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception as e: logger.error(f"Error stopping SoundDeviceInput: {e}", exc_info=True)
            finally: self._input_stream = None
        self._is_running = False
        self._is_paused = False

    def pause(self):
        if self._is_running and not self._is_paused:
            self._is_paused = True
            logger.info("SoundDeviceInput paused.")

    def resume(self):
        if self._is_running and self._is_paused:
            self._is_paused = False
            logger.info("SoundDeviceInput resumed.")
            
    async def shutdown(self): 
        self.stop()


class SoundDeviceOutputEngine(AudioOutputEngineBase):
    def __init__(self, config: Dict[str, Any]): 
        self.sd_settings: SoundDeviceSettings = config["sounddevice_settings"]
        self.audio_settings: AudioSettings = config["audio_settings"] 
        
        self._output_queue = queue.Queue() # Stores (audio_bytes, sample_rate)
        self._output_thread = None
        self._stop_event = threading.Event()
        self._is_running = False # Indicates if the worker thread is active
        self._enabled = self.sd_settings.enabled

        self.current_stream: Optional[sd.OutputStream] = None
        self.current_stream_sample_rate: Optional[int] = None

        if self._enabled:
            try:
                # Initial check with default/configured rate
                sd.check_output_settings(
                    device=self.sd_settings.output_device_index,
                    samplerate=self.sd_settings.tts_output_sample_rate, # Used for initial check
                    channels=self.audio_settings.channels 
                )
                logger.info(f"SoundDeviceOutput: Device={self.sd_settings.output_device_index}, "
                            f"Default Rate={self.sd_settings.tts_output_sample_rate}")
            except Exception as e:
                logger.error(f"SoundDeviceOutput configuration error: {e}", exc_info=True)
                self._enabled = False
        else:
            logger.info("SoundDeviceOutput is disabled by settings.")
            
    @property
    def is_enabled(self) -> bool: return self._enabled
    @property
    def is_running(self) -> bool: return self._is_running

    def _output_worker(self):
        logger.info("SoundDeviceOutput worker thread started.")
        self._is_running = True
        
        while not self._stop_event.is_set():
            try:
                item = self._output_queue.get(timeout=0.5)
                if item is None: # Sentinel to stop
                    logger.debug("SoundDeviceOutput worker received stop sentinel.")
                    break 
                
                audio_bytes, sample_rate_for_chunk = item

                if self.current_stream is None or sample_rate_for_chunk != self.current_stream_sample_rate:
                    if self.current_stream:
                        logger.info(f"SoundDeviceOutput: Changing sample rate from {self.current_stream_sample_rate}Hz to {sample_rate_for_chunk}Hz.")
                        try:
                            self.current_stream.stop()
                            self.current_stream.close()
                        except Exception as e:
                            logger.error(f"SoundDeviceOutput: Error closing existing stream: {e}")
                        self.current_stream = None
                    
                    try:
                        logger.info(f"SoundDeviceOutput: Opening new stream with SR={sample_rate_for_chunk}Hz.")
                        self.current_stream = sd.OutputStream(
                            samplerate=sample_rate_for_chunk,
                            blocksize=0, # Let sounddevice choose, or a reasonable default like 1024
                            device=self.sd_settings.output_device_index,
                            channels=self.audio_settings.channels,
                            dtype='int16' # Assuming TTS provides int16 PCM
                        )
                        self.current_stream.start()
                        self.current_stream_sample_rate = sample_rate_for_chunk
                    except Exception as e:
                        logger.error(f"SoundDeviceOutput: Failed to open stream with SR={sample_rate_for_chunk}Hz: {e}", exc_info=True)
                        self.current_stream = None
                        self.current_stream_sample_rate = None
                        self._output_queue.task_done() # Mark item as processed to avoid hang
                        continue # Skip this chunk

                if self.current_stream and self.current_stream_sample_rate == sample_rate_for_chunk:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    self.current_stream.write(audio_array)
                else:
                    logger.warning(f"SoundDeviceOutput: Stream not ready or sample rate mismatch for chunk. SR_expected={sample_rate_for_chunk}, SR_stream={self.current_stream_sample_rate}")
                
                self._output_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in SoundDeviceOutput worker loop: {e}", exc_info=True)
                time.sleep(0.1) # Avoid busy-looping on persistent errors
        
        # Cleanup stream when worker stops
        if self.current_stream:
            logger.info("SoundDeviceOutput worker: Stopping and closing final audio stream.")
            try:
                self.current_stream.stop()
                self.current_stream.close()
            except Exception as e:
                logger.error(f"SoundDeviceOutput: Error closing stream on worker exit: {e}")
            self.current_stream = None
            self.current_stream_sample_rate = None
        
        self._is_running = False
        logger.info("SoundDeviceOutput worker thread stopped.")


    def start(self):
        if not self._enabled or self._is_running: return
        logger.info("Attempting to start SoundDeviceOutput worker...")
        self._stop_event.clear()
        # Clear queue before starting
        while not self._output_queue.empty():
            try: self._output_queue.get_nowait()
            except queue.Empty: break
        self._output_thread = threading.Thread(target=self._output_worker, daemon=True, name="SoundDeviceOutputThread")
        self._output_thread.start()
        # self._is_running is set by the worker thread itself.
        # Wait a brief moment to allow the thread to start and set _is_running
        time.sleep(0.1) 
        if not self._is_running:
            logger.warning("SoundDeviceOutput worker thread may not have started correctly.")


    def stop(self):
        if not self._enabled or not self._is_running: # Check _is_running
             if self._enabled and not self._output_thread: # It was never started
                  logger.info("SoundDeviceOutput was enabled but not running, no stop action needed.")
             elif self._enabled: # It was running or attempted to run
                  logger.info("SoundDeviceOutput stop called, but worker not marked as running.")
             return

        logger.info("Attempting to stop SoundDeviceOutput worker...")
        self._stop_event.set()
        self._output_queue.put(None) # Sentinel to unblock queue.get()
        
        if self._output_thread and self._output_thread.is_alive():
             logger.debug("SoundDeviceOutput: Joining worker thread...")
             self._output_thread.join(timeout=3.0) # Increased timeout
             if self._output_thread.is_alive():
                 logger.warning("SoundDeviceOutput worker thread did not terminate gracefully.")
        
        self._output_thread = None
        # _is_running should be set to False by the worker thread itself upon exit.
        # If join times out, _is_running might remain true, indicating an issue.
        if self._is_running: # If it's still true after join
            logger.warning("SoundDeviceOutput worker thread still marked as running after stop attempt.")
            self._is_running = False # Force it now

        logger.info("SoundDeviceOutput stop sequence complete.")


    def play_tts_bytes(self, audio_bytes: bytes, sample_rate: Optional[int] = None):
        if not self._enabled:
            logger.debug("SoundDeviceOutput is disabled, cannot play TTS.")
            return
        if not self._is_running: 
            logger.warning("SoundDeviceOutput not running, attempting to start it to play TTS.")
            self.start() # Try to start if not running
            if not self._is_running: # Check again after attempting start
                 logger.error("SoundDeviceOutput failed to start, cannot play TTS.")
                 return

        if audio_bytes:
            sr_to_use = sample_rate if sample_rate is not None else self.sd_settings.tts_output_sample_rate
            logger.debug(f"SoundDeviceOutput: Queuing {len(audio_bytes)} bytes with SR={sr_to_use}Hz.")
            self._output_queue.put((audio_bytes, sr_to_use))
        else:
            logger.debug("SoundDeviceOutput: play_tts_bytes called with empty audio_bytes.")

    async def shutdown(self): 
        self.stop()