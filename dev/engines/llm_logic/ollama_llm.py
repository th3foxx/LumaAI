import logging
from typing import Union, Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_community.chat_models import ChatOllama # Ensure langchain_community is installed

from .base import LLMLogicEngineBase
from settings import OllamaSettings # Specific settings for Ollama

logger = logging.getLogger(__name__)

class OllamaLLMEngine(LLMLogicEngineBase):
    def __init__(self, config: Dict[str, Any]):
        self.settings: OllamaSettings = config.get("ollama_settings")
        if not self.settings:
            raise ValueError("OllamaLLMEngine requires 'ollama_settings' in config.")

        if not self.settings.base_url or not self.settings.model:
            logger.warning("Ollama base_url or model not configured. OllamaLLMEngine may not function.")
            self._llm: Optional[ChatOllama] = None
            return

        try:
            self._llm = ChatOllama(
                base_url=self.settings.base_url,
                model=self.settings.model,
                temperature=self.settings.temperature,
                # Add other parameters like num_ctx, top_k, top_p if needed from settings
                # e.g., options={"num_ctx": self.settings.num_ctx}
            )
            logger.info(f"ChatOllama initialized with model: {self.settings.model} at {self.settings.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOllama: {e}", exc_info=True)
            self._llm = None
            # Depending on severity, you might want to raise this error
            # raise RuntimeError(f"Could not initialize ChatOllama: {e}") from e
    
    async def ask(self, question: str, thread_id: str, return_full: bool = False) -> Union[str, BaseMessage]:
        if not self._llm:
            logger.error("Ollama LLM not initialized. Cannot process request.")
            error_content = "Sorry, the offline language model is not available."
            return AIMessage(content=error_content) if return_full else error_content

        logger.debug(f"OllamaLLMEngine: ask called for thread '{thread_id}'. Question: '{question[:50]}...'")
        
        # For simpler offline models, history might not be as critical or managed externally.
        # Here, we'll send a system prompt (if defined) and the human question.
        messages = []
        if self.settings.system_prompt:
            messages.append(SystemMessage(content=self.settings.system_prompt))
        messages.append(HumanMessage(content=question))

        try:
            response = await self._llm.ainvoke(messages) # Use ainvoke for async
            
            logger.debug(f"OllamaLLMEngine raw response: {response}")
            
            if return_full:
                return response # This will be an AIMessage
            else:
                return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"OllamaLLMEngine: Error invoking Ollama model: {e}", exc_info=True)
            error_content = f"Sorry, an error occurred with the offline AI model."
            return AIMessage(content=error_content) if return_full else error_content

    async def startup(self):
        # Check if Ollama server is reachable?
        # For now, we assume it's running if configured.
        if self._llm:
            logger.info("OllamaLLMEngine started. (Assumes Ollama server is running)")
        else:
            logger.warning("OllamaLLMEngine started, but LLM is not initialized (check config).")


    async def shutdown(self):
        # No specific shutdown needed for ChatOllama client typically
        logger.info("OllamaLLMEngine shutdown.")