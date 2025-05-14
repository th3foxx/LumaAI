import logging
import json
from typing import Dict, Any, Optional

from .base import OfflineCommandProcessorBase
from engines.communication.base import CommunicationServiceBase # For type hint

# Import helpers (assume they are in a utils module or accessible from offline_controller)
# Ensure offline_controller.py is in a location where it can be imported,
# or move these utilities to a proper utils.py file.
from utils.text_processing import find_best_match, _normalize_russian_text_with_pymorphy, \
                                  _PRE_NORMALIZED_CITY_TO_TIMEZONE, _NLU_ATTR_TO_STANDARD, \
                                  initialize_city_timezones, morph, COLOR_NAME_TO_XY, \
                                  COLOR_TEMP_PRESETS, normalize_text_key # (NEW)

# Import tool functions that can be called directly
from tools.time import get_current_time # Langchain tool, can be invoked
from tools.device_control import set_device_attribute_directly # Assumed helper for direct MQTT calls

logger = logging.getLogger(__name__)

# Ensure Pymorphy and city timezones are initialized if not already
# This is redundant if rasa_nlu.py already does it, but safe.
if morph is None:
    try:
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()
        logger.info("Pymorphy3 MorphAnalyzer initialized by DefaultOfflineCommandProcessor.")
    except Exception as e:
        logger.error(f"Failed to initialize Pymorphy3 in DefaultOfflineCommandProcessor: {e}")
        morph = None
initialize_city_timezones()


class DefaultOfflineCommandProcessor(OfflineCommandProcessorBase):
    def __init__(self, comm_service: CommunicationServiceBase):
        self.comm_service = comm_service

    async def process_nlu_result(self, nlu_result: Dict[str, Any],
                                 last_mentioned_device_context: Optional[str]
                                 ) -> Dict[str, Any]: # Returns a "resolved_command" structure
        intent = nlu_result.get("intent")
        
        # Base structure for the resolved command
        resolved_command = {
            "executable": False,
            "intent": intent,
            "nlu_raw_output": nlu_result, # Store the raw NLU output for debugging/potential use
            "response_on_failure": "Sorry, I couldn't fully understand or process that offline."
            # "resolved_device_name_for_context_update": None # Will be set if device is resolved
        }

        if not intent:
            resolved_command["response_on_failure"] = "I didn't understand the intent of your command."
            return resolved_command

        # --- GET_TIME INTENT ---
        if intent == "get_time":
            # NLU engine already extracts 'timezone_str_for_tool'
            timezone_arg = nlu_result.get("timezone_str_for_tool")
            resolved_command.update({
                "executable": True,
                "tool_to_call": "get_current_time_tool_invoke",
                "tool_args": {"timezone_str": timezone_arg}
            })
            return resolved_command

        # --- DEVICE CONTROL INTENTS ---
        elif intent in ["activate_device", "deactivate_device", "set_attribute"]:
            # These entities are provided by RasaNLUEngine based on its parsing:
            # - device_description_text: Combined text for device description + location.
            # - device_is_pronoun: Boolean indicating if a device pronoun was used.
            # - attribute: Standardized attribute name (e.g., "brightness", "state").
            # - raw_value: Normalized value for the attribute.
            
            device_description_from_nlu = nlu_result.get("device_description_text")
            is_pronoun = nlu_result.get("device_is_pronoun", False)
            
            resolved_device_name: Optional[str] = None
            available_devices = self.comm_service.get_device_friendly_names()

            if not available_devices:
                resolved_command["response_on_failure"] = "I can't see any devices to control right now."
                return resolved_command

            # Determine the text to use for fuzzy matching
            search_text_for_matching = None
            if is_pronoun and last_mentioned_device_context:
                # If NLU said it's a pronoun AND we have a context device, use the context.
                search_text_for_matching = last_mentioned_device_context
                logger.debug(f"Offline Processor: Pronoun indicated by NLU. Using context '{search_text_for_matching}' for device search.")
            elif device_description_from_nlu:
                # If NLU provided a description, use that.
                search_text_for_matching = device_description_from_nlu
            elif intent in ['set_attribute', 'activate_device', 'deactivate_device'] and last_mentioned_device_context:
                # Fallback: No specific device mentioned by NLU, but it's a device control intent, and we have context.
                search_text_for_matching = last_mentioned_device_context
                logger.debug(f"Offline Processor: No device in NLU, but device intent. Using context '{search_text_for_matching}'.")


            if search_text_for_matching:
                # Normalize the search text before fuzzy matching
                normalized_search_text = _normalize_russian_text_with_pymorphy(search_text_for_matching)
                if normalized_search_text: # Ensure not empty after normalization
                    matched_device = find_best_match(normalized_search_text, available_devices, score_cutoff=70)
                    if matched_device:
                        resolved_device_name = matched_device
                        logger.info(f"Offline Processor: Device resolved via search '{search_text_for_matching}' (norm: '{normalized_search_text}') to '{resolved_device_name}'")
                else:
                    logger.warning(f"Offline Processor: Search text '{search_text_for_matching}' became empty after normalization.")
            
            if not resolved_device_name:
                original_search_ref = device_description_from_nlu or \
                                      (last_mentioned_device_context if is_pronoun or not device_description_from_nlu else "an unspecified device")
                resolved_command["response_on_failure"] = f"I'm not sure which device you mean based on '{original_search_ref}'."
                return resolved_command

            # Attribute and Value are already processed by RasaNLUEngine
            attribute_name = nlu_result.get("attribute") # Already standardized by NLU engine
            raw_value_from_nlu = nlu_result.get("raw_value") # Already normalized by NLU engine

            if not attribute_name: # Should be set by NLU for these intents
                resolved_command["response_on_failure"] = f"I understood the device '{resolved_device_name}', but not what to change."
                return resolved_command
            if raw_value_from_nlu is None and intent == "set_attribute": # Value needed for set_attribute
                resolved_command["response_on_failure"] = f"For '{resolved_device_name}', I need a value to set for '{attribute_name}'."
                return resolved_command

            resolved_command.update({
                "executable": True,
                "device_friendly_name": resolved_device_name, # This is the resolved name
                "attribute_name": attribute_name,
                "value_str": str(raw_value_from_nlu), # Tool expects string
                "tool_to_call": "set_device_attribute_directly",
                "tool_args": {
                    "device_friendly_name": resolved_device_name,
                    "attribute_name": attribute_name,
                    "value_str": str(raw_value_from_nlu)
                },
                "resolved_device_name_for_context_update": resolved_device_name # IMPORTANT FOR CONTEXT
            })
            return resolved_command
        else:
            resolved_command["response_on_failure"] = f"I don't know how to handle the offline intent: {intent}."
            return resolved_command

    async def execute_resolved_command(self, resolved_command: Dict[str, Any]) -> str:
        # ... (this method remains largely the same as previously defined) ...
        if not resolved_command.get("executable"):
            return resolved_command.get("response_on_failure", "Command could not be executed.")

        tool_to_call = resolved_command.get("tool_to_call")
        tool_args = resolved_command.get("tool_args", {})
        
        logger.info(f"Offline Processor: Executing '{tool_to_call}' with args: {tool_args}")

        try:
            if tool_to_call == "get_current_time_tool_invoke":
                return get_current_time.invoke(tool_args) 
            elif tool_to_call == "set_device_attribute_directly":
                return await set_device_attribute_directly(
                    comm_service=self.comm_service,
                    **tool_args
                )
            else:
                logger.warning(f"Offline Processor: Unknown tool_to_call '{tool_to_call}'")
                return f"I don't know how to execute the action: {tool_to_call}."
        except Exception as e:
            logger.error(f"Offline Processor: Error executing {tool_to_call}: {e}", exc_info=True)
            return f"Sorry, an error occurred while trying to perform the action."