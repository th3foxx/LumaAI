# In sounddevice_io.py
import asyncio
import logging
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from typing import Callable, Coroutine, Any, Dict, Optional

# Импортируем нашу утилиту для ресемплинга
from utils.audio_processing import resample_audio_bytes, SCIPY_AVAILABLE

from .input_base import AudioInputEngineBase, ProcessAudioCallback
from .output_base import AudioOutputEngineBase
from settings import SoundDeviceSettings, AudioSettings

logger = logging.getLogger(__name__)

# --- SoundDeviceInputEngine остается без изменений ---
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
                dtype='float32', # Sounddevice ожидает float32 для callback
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
        self._is_paused = False # Сбрасываем паузу при полной остановке

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

        self._output_queue = queue.Queue()
        self._output_thread = None
        self._stop_event = threading.Event()
        self._is_running = False
        self._enabled = self.sd_settings.enabled

        self.output_stream: Optional[sd.OutputStream] = None
        # Частота, с которой будет работать OutputStream.
        # Если fixed_output_sample_rate задан, используется он, иначе tts_output_sample_rate как fallback.
        self.stream_working_sr: int = self.sd_settings.fixed_output_sample_rate \
                                     if self.sd_settings.fixed_output_sample_rate is not None \
                                     else self.sd_settings.tts_output_sample_rate

        if self._enabled:
            if self.sd_settings.fixed_output_sample_rate is not None and not SCIPY_AVAILABLE:
                logger.error("SoundDeviceOutput: fixed_output_sample_rate is set, but SciPy is not available for resampling. "
                             "Output may be incorrect or fail. Disabling fixed rate mode.")
                # Отключаем режим фиксированной частоты, если нет SciPy, но движок остается включенным
                # и будет пытаться работать с частотой входящих чанков.
                self.stream_working_sr = self.sd_settings.tts_output_sample_rate # Возврат к старому поведению
            try:
                logger.info(f"SoundDeviceOutput: Initializing with stream working SR: {self.stream_working_sr}Hz "
                            f"(Fixed mode: {self.sd_settings.fixed_output_sample_rate is not None and SCIPY_AVAILABLE})")
                sd.check_output_settings(
                    device=self.sd_settings.output_device_index,
                    samplerate=self.stream_working_sr,
                    channels=self.audio_settings.channels
                )
                logger.info(f"SoundDeviceOutput: Device={self.sd_settings.output_device_index}, "
                            f"Stream SR={self.stream_working_sr}Hz")
            except Exception as e:
                logger.error(f"SoundDeviceOutput configuration error with SR {self.stream_working_sr}Hz: {e}", exc_info=True)
                self._enabled = False
        else:
            logger.info("SoundDeviceOutput is disabled by settings.")

    @property
    def is_enabled(self) -> bool: return self._enabled
    @property
    def is_running(self) -> bool: return self._is_running

    def _ensure_stream_is_open(self, required_sr_for_stream_operation: int) -> bool:
        """Открывает или переоткрывает стрим, если это необходимо."""
        if self.output_stream and self.output_stream.samplerate == required_sr_for_stream_operation and not self.output_stream.closed:
            return True # Стрим уже открыт с нужной частотой

        if self.output_stream: # Закрываем старый стрим, если он есть
            logger.info(f"SoundDeviceOutput: Closing existing stream (SR={self.output_stream.samplerate}Hz).")
            try:
                if not self.output_stream.closed:
                    self.output_stream.stop()
                    self.output_stream.close()
            except Exception as e:
                logger.error(f"SoundDeviceOutput: Error closing existing stream: {e}")
            self.output_stream = None

        try:
            logger.info(f"SoundDeviceOutput: Opening new stream with SR={required_sr_for_stream_operation}Hz.")
            self.output_stream = sd.OutputStream(
                samplerate=required_sr_for_stream_operation,
                blocksize=0, # Автоматический выбор размера блока
                device=self.sd_settings.output_device_index,
                channels=self.audio_settings.channels,
                dtype='int16' # Мы всегда будем подавать int16 после ресемплинга
            )
            self.output_stream.start()
            return True
        except Exception as e:
            logger.error(f"SoundDeviceOutput: Failed to open stream with SR={required_sr_for_stream_operation}Hz: {e}", exc_info=True)
            self.output_stream = None
            return False

    def _output_worker(self):
        logger.info("SoundDeviceOutput worker thread started.")
        self._is_running = True
        
        is_fixed_sr_mode_active = self.sd_settings.fixed_output_sample_rate is not None and SCIPY_AVAILABLE
        # Эталонная/конфигурированная частота потока (для фиксированного режима или как начальная для нефиксированного)
        configured_stream_sr: int = self.stream_working_sr 

        while not self._stop_event.is_set():
            try:
                item = self._output_queue.get(timeout=0.5)
                if item is None: # Сигнал для остановки
                    logger.debug("SoundDeviceOutput worker received stop sentinel.")
                    break
                
                audio_bytes_chunk, source_sample_rate = item

                current_target_sr_for_stream: int
                bytes_to_play = audio_bytes_chunk

                if is_fixed_sr_mode_active:
                    # В фиксированном режиме целевая частота потока всегда configured_stream_sr (fixed_output_sample_rate)
                    current_target_sr_for_stream = configured_stream_sr
                    if source_sample_rate != current_target_sr_for_stream:
                        logger.debug(f"SoundDeviceOutput: Resampling chunk from {source_sample_rate}Hz to {current_target_sr_for_stream}Hz.")
                        bytes_to_play = resample_audio_bytes(
                            audio_bytes_chunk,
                            source_sample_rate,
                            current_target_sr_for_stream,
                            channels=self.audio_settings.channels,
                            dtype_str='int16' # TTS обычно int16, и resample_audio_bytes должен вернуть int16
                        )
                else: # Не фиксированный режим - поток адаптируется к частоте источника
                    current_target_sr_for_stream = source_sample_rate
                    # Ресемплинг не нужен, bytes_to_play остается audio_bytes_chunk

                # Убеждаемся, что стрим открыт с определенной целевой частотой для этого чанка
                if not self._ensure_stream_is_open(current_target_sr_for_stream):
                    logger.error(f"SoundDeviceOutput: Stream could not be opened/reopened at {current_target_sr_for_stream}Hz. Skipping chunk.")
                    self._output_queue.task_done()
                    continue

                # Поток должен быть открыт и готов
                if self.output_stream and not self.output_stream.closed:
                    if bytes_to_play: # Убедимся, что есть что играть
                        audio_array = np.frombuffer(bytes_to_play, dtype=np.int16)
                        self.output_stream.write(audio_array)
                    else:
                        logger.debug("SoundDeviceOutput: Nothing to play after resampling (empty bytes).")
                else: # Не должно произойти, если _ensure_stream_is_open отработал успешно
                    logger.warning(f"SoundDeviceOutput: Stream not ready or closed unexpectedly. Cannot play chunk.")
                
                self._output_queue.task_done()

            except queue.Empty:
                continue
            except sd.PortAudioError as pae:
                logger.error(f"SoundDeviceOutput: PortAudioError in worker loop: {pae}", exc_info=True)
                if self.output_stream: 
                    try: self.output_stream.close()
                    except: pass
                    self.output_stream = None
                time.sleep(0.1) 
            except Exception as e:
                logger.error(f"Error in SoundDeviceOutput worker loop: {e}", exc_info=True)
                time.sleep(0.1)
        
        # Очистка при завершении работы воркера (остается как было)
        if self.output_stream and not self.output_stream.closed:
            logger.info("SoundDeviceOutput worker: Stopping and closing final audio stream.")
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception as e:
                logger.error(f"SoundDeviceOutput: Error closing stream on worker exit: {e}")
        self.output_stream = None
        
        self._is_running = False
        logger.info("SoundDeviceOutput worker thread stopped.")

    def start(self):
        if not self._enabled:
            logger.info("SoundDeviceOutput is disabled, not starting worker.")
            return
        if self._is_running:
            logger.info("SoundDeviceOutput worker is already running.")
            return

        logger.info("Attempting to start SoundDeviceOutput worker...")
        self._stop_event.clear()
        # Очищаем очередь перед стартом, если там что-то осталось от предыдущих запусков
        while not self._output_queue.empty():
            try: self._output_queue.get_nowait()
            except queue.Empty: break
            self._output_queue.task_done() # Если элементы были

        self._output_thread = threading.Thread(target=self._output_worker, daemon=True, name="SoundDeviceOutputThread")
        self._output_thread.start()
        
        # Даем время потоку запуститься и установить self._is_running
        time.sleep(0.1)
        if not self._is_running:
            logger.warning("SoundDeviceOutput worker thread may not have started correctly.")

    def stop(self):
        if not self._enabled : # Если движок выключен, ничего не делаем
             logger.debug("SoundDeviceOutput is disabled, stop call ignored.")
             return
        if not self._is_running and not (self._output_thread and self._output_thread.is_alive()):
             logger.info("SoundDeviceOutput worker not running or thread not alive, no stop action needed.")
             return

        logger.info("Attempting to stop SoundDeviceOutput worker...")
        self._stop_event.set()
        # Отправляем None в очередь, чтобы разблокировать get() в воркере, если он ждет
        self._output_queue.put(None)

        if self._output_thread and self._output_thread.is_alive():
            logger.debug("SoundDeviceOutput: Joining worker thread...")
            self._output_thread.join(timeout=3.0) # Таймаут на завершение потока
            if self._output_thread.is_alive():
                logger.warning("SoundDeviceOutput worker thread did not terminate gracefully after timeout.")
            else:
                logger.info("SoundDeviceOutput worker thread joined successfully.")
        
        self._output_thread = None # Сбрасываем ссылку на поток
        # self._is_running должен быть установлен в False самим воркером при выходе.
        # Если после join он все еще True, это может указывать на проблему.
        if self._is_running:
            logger.warning("SoundDeviceOutput worker thread still marked as running after stop attempt. Forcing status.")
            self._is_running = False

        logger.info("SoundDeviceOutput stop sequence complete.")

    def play_tts_bytes(self, audio_bytes: bytes, sample_rate: Optional[int] = None):
        if not self._enabled:
            logger.debug("SoundDeviceOutput is disabled, cannot play TTS.")
            return
        
        if not self._is_running:
            logger.warning("SoundDeviceOutput not running. Attempting to start it to play TTS.")
            self.start() # Попытка запустить, если не был запущен
            if not self._is_running: # Проверяем снова после попытки запуска
                 logger.error("SoundDeviceOutput failed to start, cannot play TTS.")
                 return

        if audio_bytes:
            # Используем переданный sample_rate, или tts_output_sample_rate из настроек как fallback
            sr_of_chunk = sample_rate if sample_rate is not None else self.sd_settings.tts_output_sample_rate
            logger.debug(f"SoundDeviceOutput: Queuing {len(audio_bytes)} bytes with source SR={sr_of_chunk}Hz.")
            self._output_queue.put((audio_bytes, sr_of_chunk))
        else:
            logger.debug("SoundDeviceOutput: play_tts_bytes called with empty audio_bytes.")

    async def shutdown(self):
        self.stop()