# engines/nlu/rasa_nlu.py

import logging
import json
import httpx
from typing import Optional, Dict, Any

from .base import NLUEngineBase
from settings import RasaNLUSettings

logger = logging.getLogger(__name__)

class RasaNLUEngine(NLUEngineBase):
    def __init__(self, config: Dict[str, Any]):
        self.settings: RasaNLUSettings = config
        if not self.settings.url:
            logger.error("Rasa NLU URL not configured. RasaNLUEngine will not function.")

    async def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Sends text to the Rasa NLU server and returns the structured result.
        This engine is stateless and only responsible for NLU classification and entity extraction.
        All business logic is handled by the OfflineCommandProcessor.
        """
        if not self.settings.url:
            return None

        payload = {"text": text}
        logger.debug(f"RasaNLU parse request to {self.settings.url} for: '{text}'")

        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout) as client:
                response = await client.post(self.settings.url, json=payload)
                response.raise_for_status()
                result = response.json()
            
            logger.debug(f"Rasa NLU raw result: {json.dumps(result, indent=2, ensure_ascii=False)}")

            intent_data = result.get("intent", {})
            intent_name = intent_data.get("name")
            confidence = intent_data.get("confidence", 0.0)
            
            if not intent_name or confidence < self.settings.intent_confidence_threshold:
                logger.warning(
                    f"NLU confidence low ({confidence:.2f} < {self.settings.intent_confidence_threshold}) "
                    f"or intent not found for text: '{text}'."
                )
                return None

            logger.info(f"Rasa NLU classified intent as '{intent_name}' with confidence {confidence:.2f}")

            # Return the raw, structured data. No processing happens here.
            return {
                "intent": intent_name,
                "entities": result.get("entities", [])
            }

        except httpx.RequestError as e:
            logger.error(f"Rasa NLU connection error to {self.settings.url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Rasa NLU server error {e.response.status_code}. Response: {e.response.text[:200]}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from Rasa NLU: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Rasa NLU parsing: {e}", exc_info=True)
        
        return None