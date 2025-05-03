import os
from dataclasses import dataclass, field
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
    max_listening_frames: int = int(os.getenv("VAD_MAX_LISTENING_FRAMES", 300))


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
class ParoliServerSettings:
    """Настройки Paroli TTS."""
    encoder_path: str = os.getenv("PAROLI_ENCODER_PATH", "./models/paroli/models/encoder.onnx")
    decoder_path: Optional[str] = os.getenv("PAROLI_DECODER_PATH", "./models/paroli/models/decoder.rknn")
    config_path: Optional[str] = os.getenv("PAROLI_CONFIG_PATH", "./models/paroli/models/model.json")
    executable: str = os.getenv("PAROLI_EXECUTABLE", "./models/paroli/paroli-server")
    receive_timeout: float = 20.0 # Таймаут ожидания сообщения от Paroli (в секундах)

    ip: str = os.getenv("PAROLI_IP", "127.0.0.1")
    port: int = int(os.getenv("PAROLI_PORT", 8848))

    extra_args: List[str] = field(default_factory=list)

    @property
    def ws_url(self) -> str:
        """Вычисляет WebSocket URL на основе ip и port."""
        return f"ws://{self.ip}:{self.port}/api/v1/stream"

    speaker_id: Optional[int] = int(os.getenv("PAROLI_SPEAKER_ID")) if os.getenv("PAROLI_SPEAKER_ID") is not None else None
    audio_format: str = os.getenv("PAROLI_AUDIO_FORMAT", "pcm")
    pcm_sample_rate: int = int(os.getenv("PAROLI_PCM_SAMPLE_RATE", 22050))

    length_scale: Optional[float] = float(os.getenv("PAROLI_LENGTH_SCALE")) if os.getenv("PAROLI_LENGTH_SCALE") else None
    noise_scale: Optional[float] = float(os.getenv("PAROLI_NOISE_SCALE")) if os.getenv("PAROLI_NOISE_SCALE") else None
    noise_w: Optional[float] = float(os.getenv("PAROLI_NOISE_W")) if os.getenv("PAROLI_NOISE_W") else None


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
    history_length: int = 10
    online_mode: bool = bool(os.getenv("AI_ONLINE_MODE", True))


@dataclass(frozen=True)
class MqttBrokerSettings:
    """Настройки подключения к MQTT брокеру (Mosquitto)."""
    host: str = os.getenv("MQTT_HOST", "localhost")
    port: int = int(os.getenv("MQTT_PORT", 1883))
    username: Optional[str] = os.getenv("MQTT_USER") # None если переменная не установлена
    password: Optional[str] = os.getenv("MQTT_PASSWORD") # None если переменная не установлена
    client_id_prefix: str = os.getenv("MQTT_CLIENT_ID_PREFIX", "lumi_voice_assistant_")
    default_topic_base: str = os.getenv("MQTT_Z2M_TOPIC_BASE", "zigbee2mqtt") # Базовый топик Zigbee2MQTT


@dataclass(frozen=True)
class RasaNLUSettings:
    """Настройки подключения к Rasa NLU серверу."""
    # Полный URL к эндпоинту /model/parse
    url: Optional[str] = os.getenv("RASA_NLU_URL", "http://localhost:5005/model/parse")
    # Таймаут ожидания ответа от Rasa в секундах
    timeout: float = float(os.getenv("RASA_NLU_TIMEOUT", 5.0))
    # Минимальный порог уверенности для принятия интента
    intent_confidence_threshold: float = float(os.getenv("RASA_INTENT_CONFIDENCE_THRESHOLD", 0.75))


@dataclass(frozen=True)
class Settings:
    """Главный объект конфигурации приложения."""
    picovoice: PicovoiceSettings = field(default_factory=PicovoiceSettings)
    vad: VADSettings = field(default_factory=VADSettings)
    vosk: VoskSettings = field(default_factory=VoskSettings)
    piper: PiperSettings = field(default_factory=PiperSettings)
    paroli_server: ParoliServerSettings = field(default_factory=ParoliServerSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    webapp: WebAppSettings = field(default_factory=WebAppSettings)
    postgres: PostgresSettings = field(default_factory=PostgresSettings)
    ai: AISettings = field(default_factory=AISettings)
    mqtt_broker: MqttBrokerSettings = field(default_factory=MqttBrokerSettings)
    rasa_nlu: RasaNLUSettings = field(default_factory=RasaNLUSettings)


# Единая точка доступа к настройкам
settings = Settings()