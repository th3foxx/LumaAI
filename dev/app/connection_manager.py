"""
Содержит класс ConnectionManager, который управляет состоянием WebSocket-соединения,
обрабатывает аудиопоток и координирует работу движков (WW, VAD, STT, NLU, LLM, TTS).
"""
import asyncio
import base64
import json
import logging
import os
import struct
import wave
from typing import Optional, Dict, List

from starlette.websockets import WebSocketState, WebSocket, WebSocketDisconnect

# Project-specific imports
from settings import AudioSettings, SoundDeviceSettings, VADSettings
from connectivity import is_internet_available
from utils.audio_processing import resample_audio_bytes, SCIPY_AVAILABLE

# Engine base imports
from engines.wake_word.base import WakeWordEngineBase
from engines.vad.base import VADEngineBase
from engines.stt.base import STTEngineBase, STTEngineBase as STTRecognizerProvider
from engines.tts.base import TTSEngineBase
from engines.nlu.base import NLUEngineBase
from engines.llm_logic.base import LLMLogicEngineBase
from engines.audio_io.output_base import AudioOutputEngineBase
from engines.communication.base import CommunicationServiceBase
from engines.offline_processing.base import OfflineCommandProcessorBase
from engines.tts.hybrid_tts import HybridTTSEngine

# App-specific imports for new modular structure
from .config import settings
from . import globals as G
from .ltm_buffer import (
    LTM_BUFFER_LOCK, LTM_MESSAGE_BUFFER, MAX_BUFFER_SIZE_PER_THREAD, 
    _save_thread_messages_to_ltm, filter_messages_for_ltm
)

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self, wake_word_engine: Optional[WakeWordEngineBase],
                 vad_engine: Optional[VADEngineBase],
                 stt_provider: Optional[STTRecognizerProvider],
                 tts_engine: Optional[TTSEngineBase],
                 nlu_engine: Optional[NLUEngineBase],
                 llm_logic_engine: Optional[LLMLogicEngineBase],
                 offline_llm_logic_engine: Optional[LLMLogicEngineBase],
                 comm_service: Optional[CommunicationServiceBase],
                 offline_processor: Optional[OfflineCommandProcessorBase],
                 global_audio_settings: AudioSettings,
                 vad_processing_settings: VADSettings,
                 sound_device_settings: SoundDeviceSettings):

        self.wake_word_engine = wake_word_engine
        self.vad_engine = vad_engine
        self.stt_provider = stt_provider
        self.tts_engine = tts_engine
        self.nlu_engine = nlu_engine
        self.llm_logic_engine = llm_logic_engine
        self.offline_llm_logic_engine = offline_llm_logic_engine
        self.comm_service = comm_service
        self.offline_processor = offline_processor
        self._last_mentioned_device_for_pronoun: Optional[str] = None
        self.global_audio_settings = global_audio_settings
        self.vad_processing_settings = vad_processing_settings
        self.sd_settings = sound_device_settings

        self.websocket: Optional[WebSocket] = None
        self.stt_recognizer: Optional[STTEngineBase.RecognizerInstance] = None
        self.state: str = "disconnected"
        self.audio_buffer: bytearray = bytearray()
        self.silence_frames_count: int = 0
        self.frames_in_listening: int = 0
        self.llm_tts_task: Optional[asyncio.Task] = None
        self.is_websocket_active: bool = False
        self.local_audio_output_engine: Optional[AudioOutputEngineBase] = None

        self.estimated_noise_floor: float = self.vad_processing_settings.probability_threshold
        self.current_dynamic_vad_threshold: float = self.vad_processing_settings.probability_threshold
        self.potential_silence_frames: int = 0
        self.was_recently_voiced: bool = False

        self.activation_sound_bytes: Optional[bytes] = None
        self.activation_sound_sample_rate: Optional[int] = None
        self.activation_sound_channels: Optional[int] = None
        self.activation_sound_sampwidth: Optional[int] = None

        try:
            if self.global_audio_settings.play_activation_sound and self.global_audio_settings.activation_sound_path:
                sound_path = self.global_audio_settings.activation_sound_path
                if sound_path and os.path.exists(sound_path):
                    with wave.open(sound_path, 'rb') as wf:
                        self.activation_sound_bytes = wf.readframes(wf.getnframes())
                        self.activation_sound_sample_rate = wf.getframerate()
                        self.activation_sound_channels = wf.getnchannels()
                        self.activation_sound_sampwidth = wf.getsampwidth()
                        logger.info(f"Activation sound loaded: {sound_path}, SR={self.activation_sound_sample_rate}Hz")

                        target_sr_for_output = self.sd_settings.fixed_output_sample_rate
                        if target_sr_for_output is not None and \
                           self.activation_sound_sample_rate != target_sr_for_output and \
                           self.activation_sound_bytes:
                            if SCIPY_AVAILABLE:
                                logger.info(f"Resampling activation sound from {self.activation_sound_sample_rate}Hz to {target_sr_for_output}Hz.")
                                self.activation_sound_bytes = resample_audio_bytes(
                                    self.activation_sound_bytes,
                                    self.activation_sound_sample_rate,
                                    target_sr_for_output,
                                    channels=self.activation_sound_channels,
                                    dtype_str=f'int{self.activation_sound_sampwidth*8}'
                                )
                            else:
                                logger.warning("SciPy not available, cannot resample activation sound.")
        except Exception as e:
            logger.error(f"Failed to load or resample activation sound: {e}", exc_info=True)
            self.activation_sound_bytes = None

        if self.wake_word_engine and self.global_audio_settings.frame_length != self.wake_word_engine.frame_length:
            logger.warning(f"Global audio frame length differs from WakeWord frame length.")
        if self.vad_engine and self.global_audio_settings.frame_length != self.vad_engine.frame_length:
            logger.warning(f"Global audio frame length differs from VAD frame length.")
        if self.stt_provider:
            try:
                self.stt_recognizer = self.stt_provider.create_recognizer()
                logger.info("ConnectionManager: STT recognizer instance created during init.")
            except Exception as e:
                logger.error(f"ConnectionManager: Failed to create STT recognizer during init: {e}", exc_info=True)
        else:
            logger.warning("ConnectionManager: STT provider not available during init. STT will not work.")

    def set_local_audio_output(self, local_audio_out_engine: AudioOutputEngineBase):
        self.local_audio_output_engine = local_audio_out_engine

    async def connect(self, websocket: WebSocket):
        if self.websocket is not None and self.websocket.client_state == WebSocketState.CONNECTED:
            logger.warning("Another WebSocket client tried to connect while one is active. Rejecting.")
            await websocket.accept()
            await websocket.close(code=1008, reason="Server busy. Another client is connected.")
            return False

        await websocket.accept()
        self.websocket = websocket
        self.state = "wakeword"
        self.audio_buffer = bytearray()
        self.silence_frames_count = 0
        self.frames_in_listening = 0
        self.llm_tts_task = None
        self.is_websocket_active = True

        if G.audio_input_engine and G.audio_input_engine.is_enabled:
            logger.info("WebSocket connected, pausing local audio input.")
            G.audio_input_engine.pause()
        if G.audio_output_engine and G.audio_output_engine.is_enabled:
            logger.info("WebSocket connected, stopping any local audio output queue.")
            G.audio_output_engine.stop()

        if not self.stt_provider:
            logger.error("STT provider not available!")
            await self.send_error("Server STT error.")
            await self.disconnect(code=1011, reason="Server STT error")
            return False
        
        try:
            if not self.stt_recognizer:
                logger.warning("STT recognizer was not created in __init__, attempting now.")
                self.stt_recognizer = self.stt_provider.create_recognizer()
                logger.info("STT recognizer created on connect.")
            else:
                self.stt_recognizer.reset()
                logger.info("STT recognizer reset on connect.")
        except Exception as e:
            logger.error(f"Failed to create/reset STT recognizer: {e}", exc_info=True)
            await self.send_error("Server STT engine failure.")
            await self.disconnect(code=1011, reason="Server STT engine failure")
            return False
        
        if not self.tts_engine or not await self.tts_engine.is_healthy():
            logger.warning("TTS engine not available or not healthy! TTS might not work.")
            await self.send_status("tts_warning", "TTS engine might be unavailable.")

        logger.info("Client connected via WebSocket.")
        await self.send_status("wakeword_listening", "Waiting for wake word...")
        return True

    async def disconnect(self, code: int = 1000, reason: str = "Client disconnected"):
        self.is_websocket_active = False

        if self.llm_tts_task and not self.llm_tts_task.done():
            self.llm_tts_task.cancel()
            try: await self.llm_tts_task
            except asyncio.CancelledError: logger.info("LLM/TTS task cancelled on disconnect.")
            except Exception as e: logger.error(f"Error during LLM/TTS task cancellation on disconnect: {e}")
        self.llm_tts_task = None

        if self.websocket:
            ws_temp = self.websocket
            self.websocket = None
            if ws_temp.client_state == WebSocketState.CONNECTED:
                try:
                    await ws_temp.close(code=code, reason=reason)
                except Exception as e:
                    logger.warning(f"Error closing WebSocket: {e}")
        
        self.state = "disconnected"
        logger.info(f"WebSocket client disconnected. Code: {code}, Reason: {reason}")

        if G.audio_input_engine and G.audio_input_engine.is_enabled:
            logger.info("WebSocket disconnected, resuming local audio input.")
            if not G.audio_input_engine.is_running: G.audio_input_engine.start()
            G.audio_input_engine.resume()
            self.state = "wakeword"
            logger.info("Local audio interface active, waiting for wake word.")
        if G.audio_output_engine and G.audio_output_engine.is_enabled:
            if not G.audio_output_engine.is_running: G.audio_output_engine.start()

    async def _send_json(self, data: dict):
        if self.is_websocket_active and self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
            try:
                await self.websocket.send_json(data)
                return True
            except WebSocketDisconnect:
                await self.disconnect(reason="WebSocket disconnected during send")
                return False
            except Exception as e:
                 logger.error(f"Error sending JSON to client: {e}")
                 return False
        return False

    async def send_status(self, status_code: str, message: str):
        await self._send_json({"type": "status", "code": status_code, "message": message})

    async def send_transcript(self, transcript: str, is_final: bool):
         await self._send_json({"type": "transcript", "text": transcript, "is_final": is_final})

    async def send_tts_info(self, sample_rate: int, channels: int = 1, bit_depth: int = 16):
        if self.is_websocket_active:
            await self._send_json({"type": "tts_info", "sample_rate": sample_rate, "channels": channels, "bit_depth": bit_depth})

    async def send_tts_chunk(self, audio_chunk_b64: str):
        if self.is_websocket_active:
            await self._send_json({"type": "tts_chunk", "data": audio_chunk_b64})

    async def send_tts_finished(self):
        if self.is_websocket_active:
            await self._send_json({"type": "tts_finished"})

    async def send_error(self, error_message: str):
        logger.error(f"Sending error to client: {error_message}")
        if self.is_websocket_active:
            await self._send_json({"type": "error", "message": error_message})
    
    async def initiate_proactive_dialogue(self, trigger_message: str):
        """
        Начинает диалог по инициативе ассистента, передавая системный триггер в LLM.
        """
        if self.state != "wakeword":
            logger.warning(f"Cannot initiate proactive dialogue, manager is busy (state: {self.state}).")
            return

        logger.info(f"Initiating proactive dialogue with trigger: '{trigger_message}'")

        # Мы немедленно запускаем LLM/TTS задачу, передавая ей триггер
        # как первый "вопрос" от системы.
        # _run_llm_tts_or_offline уже умеет обрабатывать это.
        if self.llm_tts_task and not self.llm_tts_task.done():
            self.llm_tts_task.cancel()
        
        # Запускаем основной цикл обработки, как будто "система" задала вопрос.
        # Используем основной ID потока.
        self.llm_tts_task = asyncio.create_task(
            self._run_llm_tts_or_offline(trigger_message, G.ASSISTANT_THREAD_ID)
        )
    
    async def _run_llm_tts_or_offline(self, text: str, thread_id: str):
        response_text = ""
        tts_audio_bytes_for_local = bytearray()
        current_tts_sample_rate_for_playback = None 

        try:
            if self.is_websocket_active:
                await self.send_status("processing_started", "Thinking...")
            else:
                logger.info("Processing request (local audio)...")
            self.state = "processing"

            online_capable = await is_internet_available()
            attempt_online_llm = online_capable and settings.ai.online_mode and self.llm_logic_engine
            
            online_llm_succeeded = False
            if attempt_online_llm:
                logger.info("Using online LLM (LangGraph).")
                try:
                    llm_result = await self.llm_logic_engine.ask(text, thread_id=thread_id)
                    _candidate_response = llm_result.get("response")
                    messages_for_ltm_raw = llm_result.get("ltm_messages", []) # Получаем "сырые" сообщения

                    messages_for_ltm_clean = filter_messages_for_ltm(messages_for_ltm_raw)
                    
                    logger.debug(
                        f"LTM Prep: Raw messages count: {len(messages_for_ltm_raw)}, "
                        f"Clean messages count: {len(messages_for_ltm_clean)}"
                    )
                    
                    # Add messages to LTM buffer if internet and Mem0 are available
                    if messages_for_ltm_clean and G.mem0_client: # <-- Используем очищенные сообщения
                        if online_capable:
                            async with LTM_BUFFER_LOCK:
                                LTM_MESSAGE_BUFFER[thread_id].extend(messages_for_ltm_clean) # <-- Добавляем очищенные
                                logger.debug(f"Added {len(messages_for_ltm_clean)} clean messages to LTM buffer for thread '{thread_id}'.")
                                
                                if len(LTM_MESSAGE_BUFFER[thread_id]) >= MAX_BUFFER_SIZE_PER_THREAD:
                                    logger.info(f"LTM buffer for thread '{thread_id}' reached max size. Triggering immediate save.")
                                    messages_to_save_now = list(LTM_MESSAGE_BUFFER[thread_id])
                                    LTM_MESSAGE_BUFFER[thread_id].clear()
                                    asyncio.create_task(
                                        _save_thread_messages_to_ltm(G.mem0_client, thread_id, messages_to_save_now)
                                    )
                        else:
                            logger.info("Skipping adding messages to LTM buffer: no internet connection.")

                    if _candidate_response and not _candidate_response.startswith("Sorry, an error occurred"):
                        response_text = _candidate_response
                        online_llm_succeeded = True
                        logger.info("Online LLM successfully provided a response.")
                    else:
                        logger.warning(f"Online LLM returned an empty or error-like response: '{_candidate_response}'.")
                except Exception as e:
                    logger.error(f"Exception during online LLM call: {e}. Falling back to offline.", exc_info=True)
            
            if not online_llm_succeeded:
                logger.info("Proceeding with offline processing.")
                nlu_result = None
                if self.nlu_engine:
                    nlu_result = await self.nlu_engine.parse(text)

                if nlu_result and nlu_result.get("intent"):
                    if self.offline_processor:
                        resolved_command = await self.offline_processor.process_nlu_result(nlu_result, self._last_mentioned_device_for_pronoun)
                        response_text = await self.offline_processor.execute_resolved_command(resolved_command)
                        if resolved_command.get("executable") and resolved_command.get("resolved_device_name_for_context_update"):
                            self._last_mentioned_device_for_pronoun = resolved_command["resolved_device_name_for_context_update"]
                    else:
                        response_text = "I understood an offline command, but cannot process it."
                else:
                    if self.offline_llm_logic_engine:
                        logger.info("Trying offline LLM (Ollama).")
                        lumi_response_offline = await self.offline_llm_logic_engine.ask(text, thread_id=thread_id)
                        _candidate_offline_response = lumi_response_offline if isinstance(lumi_response_offline, str) else lumi_response_offline.content
                        
                        if _candidate_offline_response and not _candidate_offline_response.startswith("Sorry, an error occurred"):
                            response_text = _candidate_offline_response
                        else:
                            logger.warning(f"Offline LLM also returned an empty or error-like response: '{_candidate_offline_response}'")
                            if not response_text:
                                response_text = "The offline assistant didn't provide a response."
                    else:
                        if not response_text:
                            response_text = "I couldn't process this online, and no suitable offline action was found."
            
            if response_text and response_text.strip() == "[DO_NOTHING]":
                logger.info("LLM decided to do nothing based on the context. Aborting proactive dialogue.")
                return

            if not response_text:
                response_text = "I'm sorry, I was unable to process your request."
            
            if response_text and self.tts_engine:
                active_tts_synthesizer = self.tts_engine
                if isinstance(self.tts_engine, HybridTTSEngine):
                    active_tts_synthesizer = await self.tts_engine.get_active_engine_for_synthesis()
                
                if active_tts_synthesizer and await active_tts_synthesizer.is_healthy():
                    current_tts_sample_rate_for_playback = active_tts_synthesizer.get_output_sample_rate()
                    logger.info(f"TTS: Using {active_tts_synthesizer.__class__.__name__} with SR {current_tts_sample_rate_for_playback}Hz.")

                    if self.is_websocket_active:
                        await self.send_tts_info(sample_rate=current_tts_sample_rate_for_playback)
                        await self.send_status("speaking_started", "Speaking...")
                    else:
                        logger.info(f"Synthesizing for local playback: '{response_text[:50]}...'")
                    self.state = "speaking"

                    async for tts_chunk in active_tts_synthesizer.synthesize_stream(response_text):
                        if self.is_websocket_active:
                            await self.send_tts_chunk(base64.b64encode(tts_chunk).decode('utf-8'))
                        if self.local_audio_output_engine and self.local_audio_output_engine.is_enabled:
                            tts_audio_bytes_for_local.extend(tts_chunk)
                    
                    if self.is_websocket_active: await self.send_tts_finished()
                    
                    if tts_audio_bytes_for_local and self.local_audio_output_engine and not self.is_websocket_active:
                        self.local_audio_output_engine.play_tts_bytes(bytes(tts_audio_bytes_for_local), sample_rate=current_tts_sample_rate_for_playback)
                else:
                    logger.error("TTS: No active and healthy synthesizer found.")
                    if self.is_websocket_active: await self.send_error("Sorry, I can't speak right now.")

        except asyncio.CancelledError:
            logger.info("LLM/TTS/Offline background task cancelled.")
        except Exception as e:
            logger.error(f"Error in _run_llm_tts_or_offline: {e}", exc_info=True)
            if self.is_websocket_active: 
                await self.send_error(f"Error processing request: {str(e)}")
        finally:
            if self.state != "disconnected": 
                self.state = "wakeword"
                if self.is_websocket_active:
                    await self.send_status("wakeword_listening", "Waiting for wake word...")
                else:
                    logger.info("Local: Waiting for wake word...")
            self.llm_tts_task = None

    async def process_audio(self, audio_chunk: bytes, thread_id: str, is_local_source: bool = False):
        if self.state in ["processing", "speaking"]: return

        if not all([self.wake_word_engine, self.vad_engine, self.stt_provider, self.stt_recognizer]):
            logger.error("Core audio processing engines not ready in ConnectionManager.")
            if not is_local_source and self.is_websocket_active: await self.send_error("Server audio engines not ready.")
            return

        expected_bytes = self.global_audio_settings.frame_length * 2
        if len(audio_chunk) != expected_bytes:
            logger.debug(f"Audio chunk size mismatch: got {len(audio_chunk)}, expected {expected_bytes}.")
            return

        try:
            pcm = struct.unpack_from(f"{self.global_audio_settings.frame_length}h", audio_chunk)
        except struct.error as e:
            logger.error(f"Audio chunk unpack error: {e}. Length: {len(audio_chunk)}")
            return

        if self.state == "wakeword":
            keyword_index = self.wake_word_engine.process(pcm)
            if keyword_index >= 0:
                logger.info("Wake word detected!")

                if self.global_audio_settings.play_activation_sound:
                    if self.is_websocket_active:
                        await self.send_status("play_activation_sound_cue", "Client should play activation sound.")
                    elif self.activation_sound_bytes and self.local_audio_output_engine and self.local_audio_output_engine.is_enabled:
                        sr_for_activation_playback = self.sd_settings.fixed_output_sample_rate \
                            if self.sd_settings.fixed_output_sample_rate is not None and SCIPY_AVAILABLE \
                            else self.activation_sound_sample_rate

                        if sr_for_activation_playback:
                            self.local_audio_output_engine.play_tts_bytes(self.activation_sound_bytes, sample_rate=sr_for_activation_playback)
                
                self.state = "listening"
                self.audio_buffer.clear()
                self.silence_frames_count = 0
                self.frames_in_listening = 0
                self.stt_recognizer.reset()

                if self.is_websocket_active:
                    await self.send_status("listening_started", "Listening...")
                else:
                     logger.info("Local: Listening started...")

        elif self.state == "listening":
            self.frames_in_listening += 1
            self.audio_buffer.extend(audio_chunk)
            self.stt_recognizer.accept_waveform(audio_chunk)

            partial_transcript = json.loads(self.stt_recognizer.partial_result()).get("partial", "")
            if partial_transcript and self.is_websocket_active:
                await self.send_transcript(partial_transcript, is_final=False)

            voice_probability = self.vad_engine.process(pcm)
            is_voiced = voice_probability > self.vad_processing_settings.probability_threshold
            
            if is_voiced:
                self.silence_frames_count = 0
            else:
                self.silence_frames_count += 1

            grace_over = self.frames_in_listening >= self.vad_processing_settings.min_listening_frames
            silence_met = self.silence_frames_count >= self.vad_processing_settings.silence_frames_threshold
            max_len_met = self.frames_in_listening >= self.vad_processing_settings.max_listening_frames

            if max_len_met or (grace_over and silence_met):
                final_transcript = json.loads(self.stt_recognizer.final_result()).get("text", "").strip()
                logger.info(f"Final Transcript: '{final_transcript}'")

                if self.is_websocket_active:
                    await self.send_transcript(final_transcript, is_final=True)

                if final_transcript:
                    if self.llm_tts_task and not self.llm_tts_task.done():
                        self.llm_tts_task.cancel()
                    self.llm_tts_task = asyncio.create_task(
                        self._run_llm_tts_or_offline(final_transcript, thread_id)
                    )
                else:
                    self.state = "wakeword"
                    if self.is_websocket_active:
                        await self.send_status("wakeword_listening", "No speech detected. Waiting for wake word...")
                    else:
                        logger.info("Local: No speech detected. Waiting for wake word...")

                self.audio_buffer.clear()
                self.silence_frames_count = 0
                self.frames_in_listening = 0