from typing import Optional, Any, TYPE_CHECKING

# Project specific imports
from engines.wake_word.base import WakeWordEngineBase
from engines.vad.base import VADEngineBase
from engines.stt.base import STTEngineBase, STTEngineBase as STTRecognizerProvider
from engines.tts.base import TTSEngineBase
from engines.nlu.base import NLUEngineBase
from engines.llm_logic.base import LLMLogicEngineBase
from engines.audio_io.input_base import AudioInputEngineBase
from engines.audio_io.output_base import AudioOutputEngineBase
from engines.communication.base import CommunicationServiceBase
from engines.offline_processing.base import OfflineCommandProcessorBase

if TYPE_CHECKING:
    from .connection_manager import ConnectionManager

# Глобальные экземпляры движков
wake_word_engine: Optional[WakeWordEngineBase] = None
vad_engine: Optional[VADEngineBase] = None
stt_engine: Optional[STTRecognizerProvider] = None
tts_engine: Optional[TTSEngineBase] = None
nlu_engine: Optional[NLUEngineBase] = None
llm_logic_engine: Optional[LLMLogicEngineBase] = None
offline_llm_logic_engine: Optional[LLMLogicEngineBase] = None
audio_input_engine: Optional[AudioInputEngineBase] = None
audio_output_engine: Optional[AudioOutputEngineBase] = None
comm_service: Optional[CommunicationServiceBase] = None
offline_command_processor: Optional[OfflineCommandProcessorBase] = None
mem0_client: Optional[Any] = None

# Главный менеджер соединений
manager: Optional["ConnectionManager"] = None

# ID потока для ассистента
ASSISTANT_THREAD_ID = "lumi-voice-assistant"