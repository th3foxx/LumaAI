from abc import ABC, abstractmethod
from typing import Union, Dict, Any
from langchain_core.messages import BaseMessage # For type hinting if returning full message

class LLMLogicEngineBase(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    async def ask(self, question: str, thread_id: str, return_full: bool = False) -> Union[str, BaseMessage]:
        """
        Processes a question/statement using the LLM and associated tools.
        """
        pass

    async def startup(self):
        """Initializes resources like database connections if needed."""
        pass

    async def shutdown(self):
        """Closes resources."""
        pass