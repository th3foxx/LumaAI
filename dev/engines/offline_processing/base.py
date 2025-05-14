from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Forward declaration for type hint
if False: # TYPE_CHECKING
    from engines.communication.base import CommunicationServiceBase

class OfflineCommandProcessorBase(ABC):
    @abstractmethod
    def __init__(self, comm_service: 'CommunicationServiceBase', # type: ignore
                 # Potentially other shared resources or configs
                ):
        pass

    @abstractmethod
    async def process_nlu_result(self, nlu_result: Dict[str, Any],
                                 last_mentioned_device_context: Optional[str]
                                 ) -> Dict[str, Any]: # Returns a "resolved_command" structure
        """
        Takes raw NLU output and context, resolves devices, handles pronouns,
        and prepares a structured command for execution.
        Example output:
        {
            "executable": True, # False if resolution failed
            "intent": "set_attribute",
            "device_friendly_name": "living_room_light",
            "attribute_name": "brightness",
            "value": 50, # Processed value
            "tool_to_call": "set_device_attribute_directly", # Or some identifier
            "tool_args": {"device_friendly_name": ..., "attribute_name": ..., "value_str": ...},
            "response_on_failure": "I couldn't figure out which device."
        }
        or for get_time:
        {
            "executable": True,
            "intent": "get_time",
            "tool_to_call": "get_current_time_tool_invoke",
            "tool_args": {"timezone_str": "Europe/Moscow"},
        }
        """
        pass

    @abstractmethod
    async def execute_resolved_command(self, resolved_command: Dict[str, Any]) -> str:
        """
        Executes the command prepared by process_nlu_result.
        This method would call the appropriate tool or directly use comm_service.
        """
        pass