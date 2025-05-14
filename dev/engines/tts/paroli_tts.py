import asyncio
import base64
import json
import logging
import os
import websockets
from typing import AsyncIterator, Dict, Any

from .base import TTSEngineBase
from settings import ParoliServerSettings # Specific settings for Paroli

logger = logging.getLogger(__name__)

class ParoliTTSEngine(TTSEngineBase):
    def __init__(self, config: Dict[str, Any]): # config here is settings.paroli_server
        self.settings: ParoliServerSettings = config # Store specific settings
        self.paroli_server_process = None
        self.paroli_log_reader_task = None

        if not self.settings.executable:
             logger.warning("Path to paroli-server executable not set.")
        if not self.settings.ws_url:
             logger.warning("Paroli Server WebSocket URL is missing.")
        if self.settings.audio_format == "pcm" and not self.settings.pcm_sample_rate:
             logger.warning("Paroli Server configured for 'pcm', but pcm_sample_rate not set.")

    async def _read_stream(self, stream, log_prefix):
        while True:
            try:
                line = await stream.readline()
                if line:
                    logger.info(f"[{log_prefix}] {line.decode('utf-8', errors='ignore').strip()}")
                else:
                    break 
            except Exception as e:
                logger.error(f"Error reading stream {log_prefix}: {e}")
                break

    async def startup(self):
        """Запускает процесс paroli-server."""
        if self.paroli_server_process is not None and self.paroli_server_process.returncode is None:
            logger.warning("Paroli server process already seems to be running.")
            return

        if not self.settings.executable or not os.path.exists(self.settings.executable):
            logger.error(f"Paroli server executable not found or not set: {self.settings.executable}")
            logger.error("TTS (Paroli) will not be available.")
            return

        command = [
            self.settings.executable,
            "--encoder", self.settings.encoder_path,
            "--decoder", self.settings.decoder_path,
            "-c", self.settings.config_path,
            "--ip", self.settings.ip,
            "--port", str(self.settings.port),
        ]
        command.extend(self.settings.extra_args)

        logger.info(f"Starting paroli-server with command: {' '.join(command)}")
        try:
            self.paroli_server_process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            logger.info(f"Paroli server process started with PID: {self.paroli_server_process.pid}")

            stdout_task = asyncio.create_task(self._read_stream(self.paroli_server_process.stdout, "paroli-stdout"))
            stderr_task = asyncio.create_task(self._read_stream(self.paroli_server_process.stderr, "paroli-stderr"))
            self.paroli_log_reader_task = asyncio.gather(stdout_task, stderr_task)

            await asyncio.sleep(3.0) 

            if self.paroli_server_process.returncode is not None:
                logger.error(f"Paroli server process exited immediately with code: {self.paroli_server_process.returncode}")
                self.paroli_server_process = None
                if self.paroli_log_reader_task:
                    self.paroli_log_reader_task.cancel()
                    self.paroli_log_reader_task = None
            else:
                logger.info("Paroli server seems to be running.")

        except FileNotFoundError:
            logger.error(f"Paroli server executable not found at: {self.settings.executable}")
            self.paroli_server_process = None
        except Exception as e:
            logger.error(f"Failed to start paroli-server: {e}", exc_info=True)
            self.paroli_server_process = None

    async def shutdown(self):
        """Останавливает процесс paroli-server."""
        if self.paroli_log_reader_task:
            self.paroli_log_reader_task.cancel()
            try:
                await self.paroli_log_reader_task
            except asyncio.CancelledError:
                pass
            self.paroli_log_reader_task = None
            logger.info("Paroli log reader tasks cancelled.")

        if self.paroli_server_process and self.paroli_server_process.returncode is None:
            pid = self.paroli_server_process.pid
            logger.info(f"Stopping paroli-server process (PID: {pid})...")
            try:
                self.paroli_server_process.terminate()
                await asyncio.wait_for(self.paroli_server_process.wait(), timeout=5.0)
                logger.info(f"Paroli server process (PID: {pid}) terminated gracefully.")
            except asyncio.TimeoutError:
                logger.warning(f"Paroli server process (PID: {pid}) did not terminate gracefully, killing...")
                try:
                    self.paroli_server_process.kill()
                    await self.paroli_server_process.wait()
                    logger.info(f"Paroli server process (PID: {pid}) killed.")
                except Exception as kill_err: # Catch ProcessLookupError here too
                    logger.error(f"Error killing paroli-server process (PID: {pid}): {kill_err}")
            except Exception as term_err: # Catch ProcessLookupError here too
                logger.error(f"Error terminating paroli-server process (PID: {pid}): {term_err}")
            finally:
                self.paroli_server_process = None
        else:
            logger.info("Paroli server process not running or already stopped.")
            
    async def is_healthy(self) -> bool:
        if not self.settings.ws_url:
            return False
        if self.paroli_server_process is None or self.paroli_server_process.returncode is not None:
            logger.warning("Paroli server process not running, TTS unhealthy.")
            return False
        try:
            # Simple health check: try to connect to the WebSocket
            async with websockets.connect(self.settings.ws_url, open_timeout=1.0, close_timeout=1.0) as ws:
                # Optionally send a ping or a very simple request if supported
                # await ws.ping() 
                logger.debug("Paroli TTS health check: WebSocket connection successful.")
                return True
        except Exception as e:
            logger.warning(f"Paroli TTS health check failed: Could not connect to {self.settings.ws_url} - {e}")
            return False


    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        if not await self.is_healthy():
             logger.error("Paroli Server not healthy or not configured. Cannot synthesize TTS.")
             # yield b"" # Or raise an error
             return

        tts_success = False
        try:
            logger.info(f"Connecting to Paroli TTS for streaming: {self.settings.ws_url}")
            async with websockets.connect(self.settings.ws_url, open_timeout=5.0, close_timeout=5.0) as paroli_ws:
                logger.info(f"Connected to Paroli TTS for synthesis: '{text[:30]}...'")

                request_payload = {"text": text, "audio_format": self.settings.audio_format}
                if self.settings.speaker_id is not None: request_payload["speaker_id"] = self.settings.speaker_id
                if self.settings.length_scale is not None: request_payload["length_scale"] = self.settings.length_scale
                # ... add other params from settings ...

                await paroli_ws.send(json.dumps(request_payload))

                while True:
                    message = None
                    try:
                        message = await asyncio.wait_for(paroli_ws.recv(), timeout=self.settings.receive_timeout)
                    except asyncio.TimeoutError:
                        logger.error("Timeout waiting for message from Paroli Server.")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"Paroli Server connection closed {'OK' if tts_success else 'unexpectedly'}.")
                        break
                    except Exception as recv_err:
                        logger.error(f"Error receiving from Paroli: {recv_err}", exc_info=True)
                        break
                    
                    if isinstance(message, bytes):
                        if len(message) > 0:
                            yield message
                        else:
                            logger.warning("Received empty audio chunk from Paroli.")
                    elif isinstance(message, str):
                        try:
                            status_data = json.loads(message)
                            if status_data.get("status") == "ok":
                                tts_success = True
                                logger.info("Paroli TTS synthesis successful.")
                            else:
                                error_msg = status_data.get("message", "Unknown TTS server error")
                                logger.error(f"Paroli Server TTS failed: {error_msg}")
                        except json.JSONDecodeError:
                            logger.error(f"Could not decode final status JSON from Paroli: {message}")
                        break # End receiving loop
                    else:
                        logger.warning(f"Unexpected message type from Paroli: {type(message)}")
            
            if not tts_success:
                logger.warning("TTS stream finished without explicit 'ok' status from Paroli.")

        except websockets.exceptions.InvalidURI:
             logger.error(f"Invalid Paroli Server WebSocket URL: {self.settings.ws_url}")
        except ConnectionRefusedError:
             logger.error(f"Connection refused by Paroli Server at {self.settings.ws_url}.")
        except asyncio.TimeoutError:
             logger.error(f"Timeout connecting to Paroli Server at {self.settings.ws_url}.")
        except Exception as e:
            logger.error(f"Paroli Server TTS streaming error: {e}", exc_info=True)
        finally:
            # Ensure the async generator is properly closed if an error occurred mid-stream
            # This is implicitly handled by exiting the async for loop in the caller
            # or by the generator context being destroyed.
            # If we needed to send a sentinel, we'd do it here, but for Paroli,
            # the JSON status message or connection close signals the end.
            pass


    async def synthesize_once(self, text: str) -> bytes:
        all_audio = bytearray()
        async for chunk in self.synthesize_stream(text):
            all_audio.extend(chunk)
        return bytes(all_audio)