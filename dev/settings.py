import os
from dataclasses import dataclass, field
from urllib.parse import quote_plus
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

# Engine Choice Enums (optional, but good for validation)
# from enum import Enum
# class STTEngineChoice(str, Enum):
#     VOSK = "vosk"
# class TTSEngineChoice(str, Enum):
#     PAROLI = "paroli"
# # ... and so on for other engine types

@dataclass(frozen=True)
class EngineSelectorSettings:
    wake_word_engine: str = os.getenv("WAKE_WORD_ENGINE", "picovoice_porcupine")
    vad_engine: str = os.getenv("VAD_ENGINE", "picovoice_cobra")
    stt_engine: str = os.getenv("STT_ENGINE", "vosk") # e.g., "vosk"
    tts_engine: str = os.getenv("TTS_ENGINE", "paroli") # e.g., "paroli"
    nlu_engine: str = os.getenv("NLU_ENGINE", "rasa")   # e.g., "rasa"
    llm_logic_engine: str = os.getenv("LLM_LOGIC_ENGINE", "langgraph") # e.g., "langgraph"
    audio_input_engine: str = os.getenv("AUDIO_INPUT_ENGINE", "sounddevice")
    audio_output_engine: str = os.getenv("AUDIO_OUTPUT_ENGINE", "sounddevice")
    communication_engine: str = os.getenv("COMMUNICATION_ENGINE", "mqtt")
    offline_processor_engine: str = os.getenv("OFFLINE_PROCESSOR_ENGINE", "default")


@dataclass(frozen=True)
class PicovoiceSettings:
    """Настройки Picovoice (Porcupine & Cobra)."""
    access_key: str = os.getenv("PICOVOICE_API_KEY", "YOUR_PICOVOICE_ACCESS_KEY_HERE")
    porcupine_keywords: List[str] = field(default_factory=lambda: ["picovoice"])
    # Optional: porcupine_model_path, porcupine_library_path
    # Optional: cobra_library_path


@dataclass(frozen=True)
class VADSettings:
    """Настройки Voice Activity Detection (used by VAD engine, e.g. Cobra)."""
    probability_threshold: float = float(os.getenv("VAD_PROBABILITY_THRESHOLD", 0.6))
    silence_frames_threshold: int = int(os.getenv("VAD_SILENCE_FRAMES_THRESHOLD", 60)) # Renamed for clarity
    min_listening_frames: int = int(os.getenv("VAD_MIN_LISTENING_FRAMES", 30))
    max_listening_frames: int = int(os.getenv("VAD_MAX_LISTENING_FRAMES", 300))


@dataclass(frozen=True)
class VoskSettings:
    """Настройки Vosk ASR."""
    model_path: str = os.getenv("VOSK_MODEL_PATH", "models/vosk-model-small-ru")
    sample_rate: int = int(os.getenv("VOSK_SAMPLE_RATE", 16000))


@dataclass(frozen=True)
class ParoliServerSettings:
    """Настройки Paroli TTS."""
    encoder_path: str = os.getenv("PAROLI_ENCODER_PATH", "./models/paroli/models/encoder.onnx")
    decoder_path: Optional[str] = os.getenv("PAROLI_DECODER_PATH", "./models/paroli/models/decoder.rknn")
    config_path: Optional[str] = os.getenv("PAROLI_CONFIG_PATH", "./models/paroli/models/model.json")
    executable: str = os.getenv("PAROLI_EXECUTABLE", "./models/paroli/paroli-server")
    receive_timeout: float = 20.0

    ip: str = os.getenv("PAROLI_IP", "127.0.0.1")
    port: int = int(os.getenv("PAROLI_PORT", 8848))
    extra_args: List[str] = field(default_factory=list)

    @property
    def ws_url(self) -> str:
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
    frame_length: int = int(os.getenv("AUDIO_FRAME_LENGTH", 512)) # Samples per frame


@dataclass(frozen=True)
class WebAppSettings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))


@dataclass(frozen=True)
class PostgresSettings:
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
class AISettings: # For LLM/LangGraph
    embedding_model: str = os.getenv("EMBED_MODEL", "google_vertexai:text-multilingual-embedding-002")
    embedding_dims: int = int(os.getenv("EMBED_DIMS", 768))
    grok_model: str = os.getenv("GROK_MODEL", "grok-2-1212") # This is specific to the LangGraphLLM
    temperature: float = float(os.getenv("TEMPERATURE", 0.0))
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    history_length: int = 10
    online_mode: bool = os.getenv("AI_ONLINE_MODE", "True").lower() in ("true", "1", "t")


@dataclass(frozen=True)
class MqttBrokerSettings:
    host: str = os.getenv("MQTT_HOST", "localhost")
    port: int = int(os.getenv("MQTT_PORT", 1883))
    username: Optional[str] = os.getenv("MQTT_USER")
    password: Optional[str] = os.getenv("MQTT_PASSWORD")
    client_id_prefix: str = os.getenv("MQTT_CLIENT_ID_PREFIX", "lumi_voice_assistant_")
    default_topic_base: str = os.getenv("MQTT_Z2M_TOPIC_BASE", "zigbee2mqtt")


@dataclass(frozen=True)
class RasaNLUSettings:
    url: Optional[str] = os.getenv("RASA_NLU_URL", "http://localhost:5005/model/parse")
    timeout: float = float(os.getenv("RASA_NLU_TIMEOUT", 5.0))
    intent_confidence_threshold: float = float(os.getenv("RASA_INTENT_CONFIDENCE_THRESHOLD", 0.75))


@dataclass(frozen=True)
class SoundDeviceSettings: # For local audio I/O via sounddevice
    enabled: bool = os.getenv("LOCAL_AUDIO_ENABLED", "True").lower() in ("true", "1", "t")
    input_device_index: Optional[int] = int(os.getenv("LOCAL_AUDIO_INPUT_DEVICE")) if os.getenv("LOCAL_AUDIO_INPUT_DEVICE") else None
    output_device_index: Optional[int] = int(os.getenv("LOCAL_AUDIO_OUTPUT_DEVICE")) if os.getenv("LOCAL_AUDIO_OUTPUT_DEVICE") else None
    # Input sample rate and frame length will come from global AudioSettings
    # Output sample rate for TTS might be different, e.g., Paroli's default
    tts_output_sample_rate: int = int(os.getenv("LOCAL_AUDIO_TTS_OUTPUT_SAMPLE_RATE", 22050)) # Default to Paroli's PCM rate


@dataclass(frozen=True)
class TelegramSettings:
    bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID") # User or group chat ID


@dataclass(frozen=True)
class Settings:
    engines: EngineSelectorSettings = field(default_factory=EngineSelectorSettings)
    picovoice: PicovoiceSettings = field(default_factory=PicovoiceSettings)
    vad_config: VADSettings = field(default_factory=VADSettings) # General VAD config
    vosk: VoskSettings = field(default_factory=VoskSettings)
    paroli_server: ParoliServerSettings = field(default_factory=ParoliServerSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    webapp: WebAppSettings = field(default_factory=WebAppSettings)
    postgres: PostgresSettings = field(default_factory=PostgresSettings)
    ai: AISettings = field(default_factory=AISettings) # For LLM
    mqtt_broker: MqttBrokerSettings = field(default_factory=MqttBrokerSettings)
    rasa_nlu: RasaNLUSettings = field(default_factory=RasaNLUSettings)
    sounddevice: SoundDeviceSettings = field(default_factory=SoundDeviceSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    scheduler_db_path: str = os.getenv("SCHEDULER_DB_PATH", "reminders.db") # Make DB path configurable
    scheduler_check_interval_seconds: int = int(os.getenv("SCHEDULER_CHECK_INTERVAL_SECONDS", 30)) # How often to check for due reminders


settings = Settings()