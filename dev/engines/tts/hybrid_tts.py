import asyncio
import logging
from typing import AsyncIterator, Dict, Any, Optional

from .base import TTSEngineBase
from connectivity import is_internet_available 
from settings import settings

logger = logging.getLogger(__name__)

class HybridTTSEngine(TTSEngineBase):
    def __init__(self, config: Dict[str, Any]):
        self.online_engine: Optional[TTSEngineBase] = config.get("online_engine")
        self.offline_engine: Optional[TTSEngineBase] = config.get("offline_engine")

        if not self.online_engine:
            logger.warning("HybridTTSEngine: Online TTS engine not provided.")
        if not self.offline_engine:
            logger.warning("HybridTTSEngine: Offline TTS engine not provided.")
        if not self.online_engine and not self.offline_engine:
            logger.error("HybridTTSEngine: Neither online nor offline TTS engines are configured. TTS will not work.")

    async def startup(self):
        logger.info("HybridTTSEngine: Starting up...")
        if self.online_engine:
            logger.info("HybridTTSEngine: Starting online TTS engine...")
            await self.online_engine.startup()
        if self.offline_engine:
            logger.info("HybridTTSEngine: Starting offline TTS engine...")
            await self.offline_engine.startup()
        logger.info("HybridTTSEngine: Startup complete.")

    async def shutdown(self):
        logger.info("HybridTTSEngine: Shutting down...")
        if self.online_engine:
            logger.info("HybridTTSEngine: Shutting down online TTS engine...")
            await self.online_engine.shutdown()
        if self.offline_engine:
            logger.info("HybridTTSEngine: Shutting down offline TTS engine...")
            await self.offline_engine.shutdown()
        logger.info("HybridTTSEngine: Shutdown complete.")

    async def is_healthy(self) -> bool:
        internet_ok = await is_internet_available()
        
        online_is_healthy = False
        if self.online_engine and internet_ok:
            online_is_healthy = await self.online_engine.is_healthy()
            if online_is_healthy:
                logger.debug("HybridTTSEngine: Online TTS is healthy and internet is available.")
                return True 

        offline_is_healthy = False
        if self.offline_engine:
            offline_is_healthy = await self.offline_engine.is_healthy()
            if offline_is_healthy:
                logger.debug("HybridTTSEngine: Offline TTS is healthy.")
                return True 
        
        if internet_ok and self.online_engine and not online_is_healthy:
             logger.warning("HybridTTSEngine: Internet is available, but online TTS is not healthy.")
        if not self.offline_engine or not offline_is_healthy:
             logger.warning("HybridTTSEngine: Offline TTS is not configured or not healthy.")
        
        if not online_is_healthy and not offline_is_healthy:
            logger.error("HybridTTSEngine is unhealthy: Neither online (considering internet) nor offline TTS is available/healthy.")
        
        return online_is_healthy or offline_is_healthy

    async def get_active_engine_for_synthesis(self) -> Optional[TTSEngineBase]:
        """
        Determines which TTS engine (online or offline) should be used based on
        internet connectivity and engine health.
        Returns the chosen engine instance, or None if no suitable engine is found.
        """
        internet_ok = await is_internet_available() and settings.ai.online_mode
        
        if self.online_engine and internet_ok:
            if await self.online_engine.is_healthy():
                logger.debug("HybridTTSEngine: Selecting online TTS engine for current synthesis.")
                return self.online_engine
            else:
                logger.warning("HybridTTSEngine: Online TTS engine available but not healthy. Will try offline for current synthesis.")
        
        if self.offline_engine:
            if await self.offline_engine.is_healthy():
                logger.debug("HybridTTSEngine: Selecting offline TTS engine for current synthesis.")
                return self.offline_engine
            else:
                logger.warning("HybridTTSEngine: Offline TTS engine not healthy for current synthesis.")
        
        logger.error("HybridTTSEngine: No active and healthy TTS engine found for current synthesis.")
        return None

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        active_engine = await self.get_active_engine_for_synthesis()
        if active_engine:
            async for chunk in active_engine.synthesize_stream(text):
                yield chunk
        else:
            logger.error("HybridTTSEngine: No TTS engine available to synthesize speech stream.")
            # Yield nothing or raise an error, current behavior is to yield nothing.

    async def synthesize_once(self, text: str) -> bytes:
        active_engine = await self.get_active_engine_for_synthesis()
        if active_engine:
            return await active_engine.synthesize_once(text)
        else:
            logger.error("HybridTTSEngine: No TTS engine available to synthesize speech at once.")
            return b""

    def get_output_sample_rate(self) -> int:
        # This method returns a "best guess" or default.
        # For accurate sample rate during synthesis, one should call:
        #   active_engine = await hybrid_engine.get_active_engine_for_synthesis()
        #   if active_engine: sample_rate = active_engine.get_output_sample_rate()
        if self.online_engine:
            try:
                # Attempt to get online engine's SR, assuming it's the preferred one
                return self.online_engine.get_output_sample_rate()
            except Exception: # Catch if online_engine or its method not ready
                pass 
        if self.offline_engine:
            try:
                return self.offline_engine.get_output_sample_rate()
            except Exception:
                pass
        # Fallback if no engine is configured or their get_output_sample_rate fails
        logger.warning("HybridTTSEngine.get_output_sample_rate: Could not determine a preferred sample rate from sub-engines. Returning default 22050 Hz.")
        return 22050 