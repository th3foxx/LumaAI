# utils/audio_processing.py
import numpy as np
import logging

try:
    from scipy.signal import resample
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Логгер здесь может быть не настроен глобально, поэтому можно использовать print
    # или настроить базовый логгер, если это критично на этапе импорта.
    # print("WARNING: SciPy not available. Audio resampling will not work.")

logger = logging.getLogger(__name__) # Получаем логгер для этого модуля

def resample_audio_bytes(audio_bytes: bytes, input_sr: int, output_sr: int, channels: int = 1, dtype_str: str = 'int16') -> bytes:
    if not SCIPY_AVAILABLE:
        # Логируем здесь, так как функция вызвана, и логгер должен быть доступен
        logger.error("Cannot resample audio: SciPy library is not available.")
        return audio_bytes # Возвращаем как есть, чтобы не прерывать поток полностью

    if input_sr == output_sr:
        return audio_bytes

    try:
        # Убедимся, что dtype_str корректен
        try:
            target_dtype = np.dtype(dtype_str)
        except TypeError:
            logger.error(f"Invalid dtype_str for resampling: {dtype_str}. Falling back to 'int16'.")
            target_dtype = np.dtype('int16')

        audio_array = np.frombuffer(audio_bytes, dtype=target_dtype)

        if channels > 1:
            # Убедимся, что данные можно корректно разбить на каналы
            if len(audio_array) % channels != 0:
                logger.error(f"Audio array length {len(audio_array)} is not divisible by channels {channels}. Cannot reshape for resampling.")
                return audio_bytes # Возвращаем оригинал
            audio_array = audio_array.reshape(-1, channels)

        num_original_samples_per_channel = audio_array.shape[0]
        # Округляем до ближайшего целого, чтобы избежать ошибок с resample
        num_target_samples_per_channel = int(round(num_original_samples_per_channel * output_sr / input_sr))

        if num_target_samples_per_channel == 0:
            if num_original_samples_per_channel > 0 : # Только если были входные сэмплы
                 logger.warning(f"Resampling from {input_sr}Hz to {output_sr}Hz resulted in 0 target samples "
                               f"for {num_original_samples_per_channel} input samples. Returning empty bytes.")
            return b''


        if channels > 1:
            resampled_channels_list = []
            for i in range(channels):
                # resample работает с 1D массивами
                resampled_channel = resample(audio_array[:, i], num_target_samples_per_channel)
                resampled_channels_list.append(resampled_channel)
            # Собираем каналы обратно
            resampled_array = np.column_stack(resampled_channels_list).ravel() # ravel() для "вытягивания" в 1D
        else: # Mono
            resampled_array = resample(audio_array, num_target_samples_per_channel)

        return resampled_array.astype(target_dtype).tobytes()
    except Exception as e:
        logger.error(f"Error during audio resampling from {input_sr}Hz to {output_sr}Hz (channels={channels}, dtype={dtype_str}): {e}", exc_info=True)
        return audio_bytes # В случае ошибки возвращаем оригинальные байты