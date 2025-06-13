import asyncio
import logging
from typing import AsyncIterator, Dict, Any, Optional
from urllib.parse import urlparse

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None # type: ignore

from .base import TTSEngineBase
from settings import ApplioTTSSettings # Specific settings for Applio TTS

logger = logging.getLogger(__name__)

class ApplioTTSEngine(TTSEngineBase):
    def __init__(self, config: ApplioTTSSettings):
        self.settings: ApplioTTSSettings = config
        self.client_session: Optional[aiohttp.ClientSession] = None

        if not AIOHTTP_AVAILABLE:
            logger.error("ApplioTTSEngine: 'aiohttp' library not found. Please install it for Applio TTS to work.")
        if not self.settings.api_url:
            logger.error("ApplioTTSEngine: API URL (APPLIO_API_URL) not configured. Applio TTS will not work.")
        if not self.settings.pth_path:
            logger.warning("ApplioTTSEngine: pth_path (APPLIO_PTH_PATH) not configured. This is likely required by Applio for voice cloning.")
        # index_path is optional (can be null for Applio)

    async def startup(self):
        if not AIOHTTP_AVAILABLE:
            logger.error("ApplioTTSEngine: Cannot start, 'aiohttp' is not available.")
            return
        
        if self.client_session and not self.client_session.closed:
            await self.client_session.close() # Close existing session first
            
        timeout = aiohttp.ClientTimeout(total=self.settings.timeout_seconds)
        self.client_session = aiohttp.ClientSession(timeout=timeout)
        logger.info(f"ApplioTTSEngine: aiohttp ClientSession started. API URL: {self.settings.api_url}")

    async def shutdown(self):
        if self.client_session and not self.client_session.closed:
            await self.client_session.close()
            logger.info("ApplioTTSEngine: aiohttp ClientSession closed.")
        self.client_session = None

    async def is_healthy(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.debug("ApplioTTSEngine is unhealthy: aiohttp not available.")
            return False
        if not self.client_session or self.client_session.closed:
            logger.debug("ApplioTTSEngine is unhealthy: aiohttp client session not available or closed.")
            return False
        if not self.settings.api_url:
            logger.warning("ApplioTTSEngine is unhealthy: API URL not configured.")
            return False

        urls_to_check = [self.settings.api_url]
        
        for url_to_check in urls_to_check:
            try:
                logger.debug(f"ApplioTTSEngine: Attempting health check with HEAD to {url_to_check}")
                health_check_timeout = aiohttp.ClientTimeout(total=max(1.0, min(5.0, self.settings.timeout_seconds / 10)))
                async with self.client_session.head(url_to_check, timeout=health_check_timeout) as response:
                    if response.status < 500:
                        logger.debug(f"ApplioTTSEngine: Health check to {url_to_check} successful with status {response.status}.")
                        return True
                    else:
                        logger.warning(f"ApplioTTSEngine: Health check to {url_to_check} received server error status {response.status}.")
            except asyncio.TimeoutError:
                logger.warning(f"ApplioTTSEngine: Health check to {url_to_check} timed out.")
            except aiohttp.ClientConnectorError as e:
                logger.warning(f"ApplioTTSEngine: Health check connection error for {url_to_check}: {e}")
            except aiohttp.ClientError as e:
                logger.warning(f"ApplioTTSEngine: ClientError during health check for {url_to_check}: {e}")
            except Exception as e:
                logger.error(f"ApplioTTSEngine: Unexpected error during health check for {url_to_check}: {e}", exc_info=True)
            
            logger.debug(f"ApplioTTSEngine: Health check failed for {url_to_check}.")

        logger.warning("ApplioTTSEngine: All health check attempts failed. Engine is unhealthy.")
        return False

    async def _perform_synthesis_request(self, text: str) -> Optional[aiohttp.ClientResponse]:
        """Helper to make the POST request and return the response object if successful."""
        if not await self.is_healthy():
            return None

        if not self.client_session or self.client_session.closed: # Should be caught by is_healthy
            logger.error("ApplioTTSEngine: client_session is not available. Attempting to restart.")
            await self.startup()
            if not self.client_session or self.client_session.closed:
                logger.error("ApplioTTSEngine: Failed to recover client_session. Synthesis aborted.")
                return None
        
        payload = {
            "text": text,
            "pth_path": self.settings.pth_path,
            "index_path": self.settings.index_path,
            "edge_voice": self.settings.edge_voice,
            "pitch": self.settings.pitch,
            "index_rate": self.settings.index_rate,
            "f0_method": self.settings.f0_method,
            "export_format": self.settings.export_format,
        }

        if not payload["pth_path"]:
            logger.warning("ApplioTTSEngine: pth_path is not set. Applio synthesis might fail or use a default voice.")

        logger.info(f"ApplioTTSEngine: Synthesizing text ('{text[:30]}...') with format '{self.settings.export_format}'.")
        logger.debug(f"ApplioTTSEngine: Request payload: {payload}")
        
        try:
            # The response object must be returned to allow the caller to stream or read it.
            # The context manager for the response will be handled by the caller.
            response = await self.client_session.post(self.settings.api_url, json=payload)
            if response.status == 200:
                logger.info(f"ApplioTTSEngine: TTS API call successful (status {response.status}).")
                return response # Return the response object for processing
            else:
                error_content = await response.text()
                logger.error(f"ApplioTTSEngine: Error from API. Status: {response.status}, Response: {error_content[:500]}")
                await response.release() # Ensure connection is released on error
                return None
        except asyncio.TimeoutError:
            logger.error(f"ApplioTTSEngine: Request to {self.settings.api_url} timed out.")
            return None
        except aiohttp.ClientError as e: # More specific client errors
            logger.error(f"ApplioTTSEngine: ClientError during TTS request: {e}", exc_info=True)
            return None
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"ApplioTTSEngine: Unexpected error during TTS request: {e}", exc_info=True)
            return None


    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        response = await self._perform_synthesis_request(text)
        if response:
            try:
                async for chunk in response.content.iter_any():
                    if chunk:
                        yield chunk
                logger.debug("ApplioTTSEngine: Finished streaming response content.")
            except Exception as e:
                logger.error(f"ApplioTTSEngine: Error while streaming response content: {e}", exc_info=True)
            finally:
                await response.release() # Ensure the response is properly closed after streaming
        # If response is None or an error occurs, this generator will simply stop.

    async def synthesize_once(self, text: str) -> bytes:
        response = await self._perform_synthesis_request(text)
        if response:
            try:
                audio_data = await response.read()
                logger.debug(f"ApplioTTSEngine: Read {len(audio_data)} bytes for non-streamed response.")
                return audio_data
            except Exception as e:
                logger.error(f"ApplioTTSEngine: Error while reading full response content: {e}", exc_info=True)
                return b""
            finally:
                await response.release() # Ensure the response is properly closed
        return b""

    def get_output_sample_rate(self) -> int:
        return self.settings.sample_rate