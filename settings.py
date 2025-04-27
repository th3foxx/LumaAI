import os
from dataclasses import dataclass
from urllib.parse import quote_plus
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()


@dataclass(frozen=True)
class PicovoiceSettings:
    """Настройки Picovoice (Porcupine)."""
    access_key: str = os.getenv("PICOVOICE_API_KEY", "YOUR_PICOVOICE_ACCESS_KEY_HERE")


@dataclass(frozen=True)
class VADSettings:
    """Настройки Voice Activity Detection."""
    probability_threshold: float = float(os.getenv("VAD_PROBABILITY_THRESHOLD", 0.6))
    silence_frames: int = int(os.getenv("VAD_SILENCE_FRAMES", 60))
    min_listening_frames: int = int(os.getenv("VAD_MIN_LISTENING_FRAMES", 30))


@dataclass(frozen=True)
class VoskSettings:
    """Настройки Vosk ASR."""
    model_path: str = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-small-ru")
    sample_rate: int = int(os.getenv("VOSK_SAMPLE_RATE", 16000))


@dataclass(frozen=True)
class PiperSettings:
    """Настройки Piper TTS."""
    model_path: str = os.getenv("PIPER_MODEL_PATH", "./models/piper/ru_RU-irina-medium.onnx")
    config_path: Optional[str] = os.getenv("PIPER_CONFIG_PATH", "./models/piper/ru_RU-irina-medium.onnx.json")
    executable: str = os.getenv("PIPER_EXECUTABLE", "./models/piper/piper")


@dataclass(frozen=True)
class AudioSettings:
    """Общие аудио-настройки."""
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))
    channels: int = int(os.getenv("AUDIO_CHANNELS", 1))
    frame_length: int = int(os.getenv("AUDIO_FRAME_LENGTH", 512))


@dataclass(frozen=True)
class WebAppSettings:
    """Параметры запуска веб-приложения."""
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))


@dataclass(frozen=True)
class PostgresSettings:
    """Параметры подключения к Postgres."""
    user: str = os.getenv("POSTGRES_USER", "lumi")
    password: str = os.getenv("POSTGRES_PASSWORD", "")
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", 5432))
    db: str = os.getenv("POSTGRES_DB", "lumi")

    @property
    def uri(self) -> str:
        pwd = quote_plus(self.password)
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class AISettings:
    """Настройки моделей встраивания и генерации."""
    embedding_model: str = os.getenv("EMBED_MODEL", "google_vertexai:text-multilingual-embedding-002")
    embedding_dims: int = int(os.getenv("EMBED_DIMS", 768))
    grok_model: str = os.getenv("GROK_MODEL", "grok-2-1212")
    temperature: float = float(os.getenv("TEMPERATURE", 0.0))
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")


@dataclass(frozen=True)
class Settings:
    """Главный объект конфигурации приложения."""
    picovoice: PicovoiceSettings = PicovoiceSettings()
    vad: VADSettings = VADSettings()
    vosk: VoskSettings = VoskSettings()
    piper: PiperSettings = PiperSettings()
    audio: AudioSettings = AudioSettings()
    webapp: WebAppSettings = WebAppSettings()
    postgres: PostgresSettings = PostgresSettings()
    ai: AISettings = AISettings()


# Единая точка доступа к настройкам
settings = Settings()
