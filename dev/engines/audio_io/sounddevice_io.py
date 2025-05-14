import asyncio
import logging
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from typing import Callable, Coroutine, Any, Dict

from .input_base import AudioInputEngineBase, ProcessAudioCallback
from .output_base import AudioOutputEngineBase
from settings import SoundDeviceSettings, AudioSettings # Specific settings

logger = logging.getLogger(__name__)

class SoundDeviceInputEngine(AudioInputEngineBase):
    def __init__(self, process_audio_callback: ProcessAudioCallback, loop: Any, 
                 config: Dict[str, Any]): # config has sounddevice_settings, audio_settings
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
            
    async def shutdown(self): # Ensure sync stop is called
        self.stop()


class SoundDeviceOutputEngine(AudioOutputEngineBase):
    def __init__(self, config: Dict[str, Any]): # config has sounddevice_settings
        self.sd_settings: SoundDeviceSettings = config["sounddevice_settings"]
        self.audio_settings: AudioSettings = config["audio_settings"] # For channels
        
        self._output_queue = queue.Queue()
        self._output_thread = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._enabled = self.sd_settings.enabled

        if self._enabled:
            try:
                sd.check_output_settings(
                    device=self.sd_settings.output_device_index,
                    samplerate=self.sd_settings.tts_output_sample_rate,
                    channels=self.audio_settings.channels # Assuming mono for now
                )
                logger.info(f"SoundDeviceOutput: Device={self.sd_settings.output_device_index}, "
                            f"Rate={self.sd_settings.tts_output_sample_rate}")
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
        stream = None
        try:
            stream = sd.OutputStream(
                samplerate=self.sd_settings.tts_output_sample_rate,
                blocksize=1024, 
                device=self.sd_settings.output_device_index,
                channels=self.audio_settings.channels,
                dtype='int16'
            )
            stream.start()
            logger.info("SoundDeviceOutput stream started.")
            self._is_running = True
            while not self._stop_event.is_set():
                try:
                    audio_bytes = self._output_queue.get(timeout=0.5)
                    if audio_bytes is None: break
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    stream.write(audio_array)
                    self._output_queue.task_done()
                except queue.Empty: continue
                except Exception as e: logger.error(f"Error in SoundDeviceOutput worker: {e}", exc_info=True); time.sleep(0.1)
        except Exception as e:
            logger.error(f"Failed to start/run SoundDeviceOutput stream: {e}", exc_info=True)
        finally:
            if stream:
                try:
                    if not stream.stopped: stream.stop()
                    stream.close()
                except Exception as e: logger.error(f"Error closing SoundDeviceOutput stream: {e}", exc_info=True)
            self._is_running = False
            self._output_queue.put(None) # Ensure any waiters unblock

    def start(self):
        if not self._enabled or self._is_running: return # Changed from self._output_thread
        logger.info("Starting SoundDeviceOutput worker...")
        self._stop_event.clear()
        while not self._output_queue.empty():
            try: self._output_queue.get_nowait()
            except queue.Empty: break
        self._output_thread = threading.Thread(target=self._output_worker, daemon=True)
        self._output_thread.start()

    def stop(self):
        if not self._enabled or not self._is_running: return # Changed from self._output_thread
        logger.info("Stopping SoundDeviceOutput worker...")
        self._stop_event.set()
        self._output_queue.put(None) 
        if self._output_thread and self._output_thread.is_alive():
             self._output_thread.join(timeout=2.0)
             if self._output_thread.is_alive(): logger.warning("SoundDeviceOutput thread didn't terminate.")
        self._output_thread = None
        self._is_running = False # Set here

    def play_tts_bytes(self, audio_bytes: bytes):
        if not self._enabled or not self._is_running: # Check _is_running
            if not self._is_running: logger.warning("SoundDeviceOutput not running, cannot play TTS.")
            return
        if audio_bytes: self._output_queue.put(audio_bytes)

    async def shutdown(self): # Ensure sync stop is called
        self.stop()