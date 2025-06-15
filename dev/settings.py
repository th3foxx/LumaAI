import os
import json
import logging
from dataclasses import dataclass, field
from urllib.parse import quote_plus
from dotenv import load_dotenv
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

load_dotenv()


def _load_system_prompt() -> str:
    """Loads the system prompt from a file path or directly from an environment variable."""
    default_prompt = "You are a helpful AI assistant."
    prompt_path = os.getenv("AI_SYSTEM_PROMPT_PATH")
    
    if prompt_path:
        try:
            # Убедимся, что путь корректный относительно корня проекта
            # Этот код предполагает, что settings.py находится в корне или подпапке
            # и вы запускаете приложение из корня проекта.
            full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', prompt_path)
            # Для более надежного определения корня, если структура сложная,
            # можно использовать `os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))`
            # но для вашей структуры это должно сработать.
            # Простой вариант, если вы всегда запускаете из корня:
            if not os.path.exists(prompt_path):
                 logger.error(f"System prompt file not found at path specified in AI_SYSTEM_PROMPT_PATH: {prompt_path}")
                 # Пробуем найти относительно файла settings.py
                 script_dir_path = os.path.join(os.path.dirname(__file__), prompt_path)
                 if os.path.exists(script_dir_path):
                     prompt_path = script_dir_path
                     logger.info(f"Found prompt file relative to settings.py: {prompt_path}")
                 else:
                     return os.getenv("AI_SYSTEM_PROMPT", default_prompt)

            with open(prompt_path, 'r', encoding='utf-8') as f:
                logger.info(f"Loading system prompt from file: {prompt_path}")
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load system prompt from file '{prompt_path}': {e}. Falling back to default or AI_SYSTEM_PROMPT variable.")
            # Если файл не найден, пробуем прочитать переменную напрямую
            return os.getenv("AI_SYSTEM_PROMPT", default_prompt)
    
    # Если путь не указан, читаем переменную как обычно
    return os.getenv("AI_SYSTEM_PROMPT", default_prompt)


@dataclass(frozen=True)
class EngineSelectorSettings:
    wake_word_engine: str = os.getenv("WAKE_WORD_ENGINE", "picovoice_porcupine")
    vad_engine: str = os.getenv("VAD_ENGINE", "picovoice_cobra")
    stt_engine: str = os.getenv("STT_ENGINE", "vosk") # e.g., "vosk"
    tts_engine: str = os.getenv("TTS_ENGINE", "hybrid") # Default to hybrid
    # Names of specific implementations for online/offline roles if tts_engine is "hybrid"
    tts_online_provider: str = os.getenv("TTS_ONLINE_PROVIDER", "applio") # e.g., "gemini"
    tts_offline_provider: str = os.getenv("TTS_OFFLINE_PROVIDER", "paroli") # e.g., "paroli"
    nlu_engine: str = os.getenv("NLU_ENGINE", "rasa")   # e.g., "rasa"
    llm_logic_engine: str = os.getenv("LLM_LOGIC_ENGINE", "langgraph") # e.g., "langgraph"
    offline_llm_engine: str = os.getenv("OFFLINE_LLM_ENGINE", "ollama") # e.g., "ollama" for offline
    audio_input_engine: str = os.getenv("AUDIO_INPUT_ENGINE", "sounddevice")
    audio_output_engine: str = os.getenv("AUDIO_OUTPUT_ENGINE", "sounddevice")
    communication_engine: str = os.getenv("COMMUNICATION_ENGINE", "mqtt")
    offline_processor_engine: str = os.getenv("OFFLINE_PROCESSOR_ENGINE", "default")


@dataclass(frozen=True)
class PicovoiceSettings:
    """Настройки Picovoice (Porcupine & Cobra)."""
    access_key: str = os.getenv("PICOVOICE_API_KEY", "YOUR_PICOVOICE_ACCESS_KEY_HERE")
    porcupine_keywords: List[str] = field(default_factory=lambda: ["picovoice"])
    keyword_paths: List[str] = field(default_factory=lambda: ["./models/Lumia_en_raspberry-pi_v3_0_0.ppn"])
    # Optional: porcupine_model_path, porcupine_library_path
    # Optional: cobra_library_path


@dataclass(frozen=True)
class VADSettings:
    """Настройки Voice Activity Detection (used by VAD engine, e.g. Cobra)."""
    probability_threshold: float = float(os.getenv("VAD_PROBABILITY_THRESHOLD", 0.6))
    silence_frames_threshold: int = int(os.getenv("VAD_SILENCE_FRAMES_THRESHOLD", 60)) # Renamed for clarity
    min_listening_frames: int = int(os.getenv("VAD_MIN_LISTENING_FRAMES", 30))
    max_listening_frames: int = int(os.getenv("VAD_MAX_LISTENING_FRAMES", 300))

    # Новые параметры для адаптивного VAD и "залипания"
    dynamic_threshold_enabled: bool = os.getenv("VAD_DYNAMIC_THRESHOLD_ENABLED", "True").lower() in ("true", "1", "t")
    noise_floor_alpha: float = float(os.getenv("VAD_NOISE_FLOOR_ALPHA", 0.05)) # Коэффициент сглаживания для оценки шума
    min_dynamic_threshold: float = float(os.getenv("VAD_MIN_DYNAMIC_THRESHOLD", 0.25)) # Мин. значение адаптивного порога
    max_dynamic_threshold: float = float(os.getenv("VAD_MAX_DYNAMIC_THRESHOLD", 0.75)) # Макс. значение адаптивного порога
    threshold_margin_factor: float = float(os.getenv("VAD_THRESHOLD_MARGIN_FACTOR", 0.15)) # Насколько порог должен быть выше оцененного шума

    speech_hangover_frames: int = int(os.getenv("VAD_SPEECH_HANGOVER_FRAMES", 8)) # Кол-во кадров "залипания" после окончания речи


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
class GeminiTTSSettings:
    """Настройки Gemini Speech TTS."""
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    # Model name for TTS. Example from prompt: "gemini-2.5-flash-preview-tts"
    model: str = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts") # Defaulting to a common pattern, user should verify/set
    voice_name: str = os.getenv("GEMINI_TTS_VOICE", "Kore") # Default from user prompt
    sample_rate: int = 24000 # Gemini TTS output is 24kHz


@dataclass(frozen=True)
class ApplioTTSSettings:
    """Настройки Applio RVC TTS. https://github.com/IAHispano/Applio"""
    api_url: str = os.getenv("APPLIO_API_URL", "http://applio.vo1dy.tech:5566/synthesize")
    # Path to the .pth model file, accessible by the Applio server
    pth_path: Optional[str] = os.getenv("APPLIO_PTH_PATH")
    # Path to the .index file, accessible by the Applio server (can be null/None)
    index_path: Optional[str] = os.getenv("APPLIO_INDEX_PATH")
    # Base voice model used by Applio, e.g., from EdgeTTS or other sources
    edge_voice: str = os.getenv("APPLIO_EDGE_VOICE", "ru-RU-SvetlanaNeural") # Example from curl
    pitch: int = int(os.getenv("APPLIO_PITCH", 2)) # Voice pitch adjustment
    index_rate: float = float(os.getenv("APPLIO_INDEX_RATE", 0.1)) # Feature retrieval ratio for .index file
    f0_method: str = os.getenv("APPLIO_F0_METHOD", "rmvpe") # Pitch estimation algorithm
    export_format: str = os.getenv("APPLIO_EXPORT_FORMAT", "WAV") # Recommended: WAV for PCM-like data. Other: MP3, OGG etc.
    # Expected sample rate of the output audio. Adjust if Applio model outputs differently.
    # For WAV, this should match the WAV header's sample rate.
    sample_rate: int = int(os.getenv("APPLIO_SAMPLE_RATE", 40000))
    timeout_seconds: float = float(os.getenv("APPLIO_TIMEOUT_SECONDS", 30.0)) # Timeout for HTTP requests to Applio


@dataclass(frozen=True)
class AudioSettings:
    """Общие аудио-настройки."""
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", 16000))
    channels: int = int(os.getenv("AUDIO_CHANNELS", 1))
    frame_length: int = int(os.getenv("AUDIO_FRAME_LENGTH", 512)) # Samples per frame
    play_activation_sound: bool = os.getenv("PLAY_ACTIVATION_SOUND", "True").lower() in ("true", "1", "t")
    activation_sound_path: Optional[str] = os.getenv("ACTIVATION_SOUND_PATH", "sounds/activation.wav")


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
class Mem0Settings:
    """Настройки для Mem0 (сервис памяти)."""
    # Общие настройки
    enabled: bool = os.getenv("MEM0_ENABLED", "True").lower() in ("true", "1", "t")
    
    # --- Ключи API ---
    # Ключ для OpenRouter
    openrouter_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "")
    # Ключ для Gemini/Google (может понадобиться для эмбеддера)
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    # --- Настройки LLM для mem0 ---
    # Модель указывается с префиксом, как в LiteLLM. 
    # Например: "openrouter/meta-llama/llama-3.1-70b-instruct"
    # или "gemini/gemini-1.5-pro"
    llm_model: str = os.getenv("MEM0_LLM_MODEL", "openrouter/meta-llama/llama-3.1-70b-instruct")
    
    # --- Настройки Embedder для mem0 ---
    # Провайдер эмбеддера (например, "gemini", "openai", "huggingface")
    embedder_provider: str = os.getenv("MEM0_EMBEDDER_PROVIDER", "gemini")
    # Модель эмбеддера
    embedder_model: str = os.getenv("MEM0_EMBEDDER_MODEL", "models/text-embedding-004")
    embedding_dims: int = int(os.getenv("MEM0_EMBEDDING_DIMS", 768))

    # --- Настройки баз данных ---
    # Настройки графовой БД (Neo4j)
    neo4j_url: Optional[str] = os.getenv("NEO4J_URL", "neo4j://localhost:7687")
    neo4j_user: Optional[str] = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: Optional[str] = os.getenv("NEO4J_PASSWORD")

    # Настройки векторной БД (Qdrant)
    qdrant_url: Optional[str] = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "lumi_memory")


@dataclass(frozen=True)
class ProactiveSettings:
    """Настройки для проактивного режима ассистента."""
    # Устройство, которое будет проверяться для сценария "вечерний свет"
    main_evening_light: str = os.getenv("PROACTIVE_MAIN_EVENING_LIGHT", "свет в лампа-зал")
    # Можно добавить другие настройки в будущем, например, включение/выключение отдельных триггеров


@dataclass(frozen=True)
class AISettings: # For LLM/LangGraph (Online)
    embedding_model: str = os.getenv("EMBED_MODEL", "google_vertexai:text-multilingual-embedding-002")
    embedding_dims: int = int(os.getenv("EMBED_DIMS", 768))
    grok_model: str = os.getenv("GROK_MODEL", "grok-2-1212") # This is specific to the LangGraphLLM
    temperature: float = float(os.getenv("TEMPERATURE", 0.0))
    openai_api_base: str = os.getenv("OPENAI_API_BASE", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    history_length: int = int(os.getenv("AI_HISTORY_LENGTH", 10))
    system_prompt: str = field(default_factory=_load_system_prompt, repr=False)
    online_mode: bool = os.getenv("AI_ONLINE_MODE", "True").lower() in ("true", "1", "t")


@dataclass(frozen=True)
class OllamaSettings: # For Offline LLM
    base_url: Optional[str] = os.getenv("OLLAMA_BASE_URL", "http://localhost:8082")
    model: Optional[str] = os.getenv("OLLAMA_MODEL", "qwen:1b") # Or your preferred model
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", 0.7))
    system_prompt: Optional[str] = os.getenv("OLLAMA_SYSTEM_PROMPT", "You are Lumi (female gender), an efficient AI voice assistant. Answer in voice AI format (without * and such).")


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
    _fixed_sr_env = os.getenv("LOCAL_AUDIO_FIXED_OUTPUT_SAMPLE_RATE")
    fixed_output_sample_rate: Optional[int] = None
    if _fixed_sr_env and _fixed_sr_env.strip(): # Check if set and not just whitespace
        try:
            fixed_output_sample_rate = int(_fixed_sr_env)
        except ValueError:
            logger.warning(
                f"Invalid integer value for LOCAL_AUDIO_FIXED_OUTPUT_SAMPLE_RATE: '{_fixed_sr_env}'. "
                "Fixed output sample rate mode will be disabled (engine will use incoming TTS sample rate or resample dynamically)."
            )


@dataclass(frozen=True)
class TelegramSettings:
    bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    
    # Настройки для Client API (отправка от вашего имени)
    client_api_id: Optional[int] = int(os.getenv("TELEGRAM_CLIENT_API_ID")) if os.getenv("TELEGRAM_CLIENT_API_ID") else None
    client_api_hash: Optional[str] = os.getenv("TELEGRAM_CLIENT_API_HASH")
    client_session_name: str = os.getenv("TELEGRAM_CLIENT_SESSION_NAME", "my_assistant_session") # Имя файла сессии
    # Номер телефона, с которого будет идти отправка.
    # Может потребоваться для первой авторизации, если сессия не найдена.
    # Можно также запрашивать интерактивно при первом запуске.
    client_phone_number: Optional[str] = os.getenv("TELEGRAM_CLIENT_PHONE_NUMBER")


@dataclass(frozen=True)
class ToolsSettings:
    """Настройки инструментов."""
    weather_api_key: str = os.getenv("WEATHER_API_KEY", "YOUR_WEATHER_API_KEY_HERE")
    weather_api_url: str = os.getenv("WEATHER_API_URL", "https://api.openweathermap.org/data/2.5/weather")
    serper_api_url: str = os.getenv("SERPER_API_URL", "https://google.serper.dev/search")
    serper_api_key: str = os.getenv("SERPER_API_KEY", "YOUR_SERPER_API_KEY_HERE")
    default_weather_location: str = os.getenv("DEFAULT_WEATHER_LOCATION", "Ейск")


@dataclass(frozen=True)
class SMTPMailSettings:
    host: Optional[str] = os.getenv("SMTP_HOST")
    port: int = int(os.getenv("SMTP_PORT", 587)) # 587 для TLS, 465 для SSL
    username: Optional[str] = os.getenv("SMTP_USERNAME")
    password: Optional[str] = os.getenv("SMTP_PASSWORD")
    use_tls: bool = os.getenv("SMTP_USE_TLS", "True").lower() in ("true", "1", "t")
    # sender_email: От чьего имени слать. Если совпадает с username, можно не указывать отдельно.
    # Если отличается, то ваш SMTP сервер должен разрешать отправку от этого имени.
    sender_email: Optional[str] = os.getenv("SMTP_SENDER_EMAIL", os.getenv("SMTP_USERNAME"))


@dataclass(frozen=True)
class ContactSettings:
    contacts_file_path: str = os.getenv("CONTACTS_FILE_PATH", "contacts.json")
    _contacts_data: Dict[str, Dict[str, str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        contacts = {}
        if self.contacts_file_path:
            try:
                with open(self.contacts_file_path, 'r', encoding='utf-8') as f:
                    loaded_contacts_raw = json.load(f)
                    # Приводим ключи (имена контактов) к нижнему регистру при загрузке
                    # для регистронезависимого поиска
                    contacts = {str(k).lower(): v for k, v in loaded_contacts_raw.items()}
                logger.info(f"Contacts loaded successfully from {self.contacts_file_path}. Found {len(contacts)} contacts.")
            except FileNotFoundError:
                logger.warning(f"Contacts file not found: {self.contacts_file_path}. No contacts will be available.")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from contacts file: {self.contacts_file_path}. Contacts not loaded.")
            except Exception as e:
                logger.error(f"Failed to load contacts from {self.contacts_file_path}: {e}. Contacts not loaded.")
        else:
            logger.warning("No contacts file path specified (CONTACTS_FILE_PATH). Contacts list will be empty.")
        # Используем object.__setattr__ так как датакласс frozen=True
        object.__setattr__(self, '_contacts_data', contacts)

    def get_contact(self, name: str) -> Optional[Dict[str, str]]:
        return self._contacts_data.get(name.lower())

    @property
    def available_contacts(self) -> List[str]:
        return list(self._contacts_data.keys())


@dataclass(frozen=True)
class Settings:
    engines: EngineSelectorSettings = field(default_factory=EngineSelectorSettings)
    picovoice: PicovoiceSettings = field(default_factory=PicovoiceSettings)
    vad_config: VADSettings = field(default_factory=VADSettings) # General VAD config
    vosk: VoskSettings = field(default_factory=VoskSettings)
    paroli_server: ParoliServerSettings = field(default_factory=ParoliServerSettings)
    gemini_tts: GeminiTTSSettings = field(default_factory=GeminiTTSSettings) # Added Gemini TTS settings
    applio_tts: ApplioTTSSettings = field(default_factory=ApplioTTSSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    webapp: WebAppSettings = field(default_factory=WebAppSettings)
    postgres: PostgresSettings = field(default_factory=PostgresSettings)
    ai: AISettings = field(default_factory=AISettings) # For LLM
    ollama: OllamaSettings = field(default_factory=OllamaSettings) # For Offline LLM
    mqtt_broker: MqttBrokerSettings = field(default_factory=MqttBrokerSettings)
    rasa_nlu: RasaNLUSettings = field(default_factory=RasaNLUSettings)
    sounddevice: SoundDeviceSettings = field(default_factory=SoundDeviceSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    tools: ToolsSettings = field(default_factory=ToolsSettings)
    scheduler_db_path: str = os.getenv("SCHEDULER_DB_PATH", "reminders.db") # Make DB path configurable
    music_db_path: str = os.getenv("MUSIC_DB_PATH", "music_data.db")
    scheduler_check_interval_seconds: int = int(os.getenv("SCHEDULER_CHECK_INTERVAL_SECONDS", 30)) # How often to check for due reminders
    smtp_mail: SMTPMailSettings = field(default_factory=SMTPMailSettings)
    contacts_config: ContactSettings = field(default_factory=ContactSettings)
    mem0: Mem0Settings = field(default_factory=Mem0Settings)
    proactive: ProactiveSettings = field(default_factory=ProactiveSettings)


settings = Settings()