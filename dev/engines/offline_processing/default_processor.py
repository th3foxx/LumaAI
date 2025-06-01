import logging
import json
from typing import Dict, Any, Optional

from .base import OfflineCommandProcessorBase
from engines.communication.base import CommunicationServiceBase 
from utils.text_processing import find_best_match, _normalize_russian_text_with_pymorphy # Removed unused imports
from tools.time import get_current_time
from tools.device_control import set_device_attribute_directly

logger = logging.getLogger(__name__)

# Pymorphy and city timezones should be initialized by RasaNLUEngine or a central app setup.
# If this module can be run independently, keep the initialization here as a fallback.
# from utils.text_processing import initialize_city_timezones, morph
# if morph is None:
#     try:
#         import pymorphy3
#         morph = pymorphy3.MorphAnalyzer()
#         logger.info("Pymorphy3 MorphAnalyzer initialized by DefaultOfflineCommandProcessor (fallback).")
#     except Exception as e:
#         logger.error(f"Failed to initialize Pymorphy3 in DefaultOfflineCommandProcessor (fallback): {e}")
#         morph = None
# initialize_city_timezones()


class DefaultOfflineCommandProcessor(OfflineCommandProcessorBase):
    def __init__(self, comm_service: CommunicationServiceBase):
        self.comm_service = comm_service
        if not comm_service:
            logger.error("DefaultOfflineCommandProcessor initialized without a CommunicationService!")

    async def process_nlu_result(self, nlu_result: Dict[str, Any],
                                 last_mentioned_device_context: Optional[str]
                                 ) -> Dict[str, Any]:
        intent = nlu_result.get("intent")
        
        resolved_command = {
            "executable": False,
            "intent": intent,
            "nlu_raw_output": nlu_result,
            "response_on_failure": "Простите, я не смогла полностью понять или обработать вашу команду в оффлайн режиме.",
            "resolved_device_name_for_context_update": None # Initialize here
        }

        if not intent:
            resolved_command["response_on_failure"] = "Простите, я не поняла, что вы хотите сделать."
            return resolved_command

        # --- GET_TIME INTENT ---
        if intent == "get_time":
            timezone_arg = nlu_result.get("timezone_str_for_tool") # NLU provides this
            resolved_command.update({
                "executable": True,
                "tool_to_call": "get_current_time_tool_invoke",
                "tool_args": {"timezone_str": timezone_arg}
                # No device context update for get_time
            })
            return resolved_command

        # --- DEVICE CONTROL INTENTS ---
        elif intent in ["activate_device", "deactivate_device", "set_attribute"]:
            device_description_from_nlu = nlu_result.get("device_description_text")
            is_pronoun = nlu_result.get("device_is_pronoun", False)
            
            resolved_device_name: Optional[str] = None
            
            if not self.comm_service or not self.comm_service.is_connected:
                resolved_command["response_on_failure"] = "Сервис управления устройствами недоступен, не могу обработать команду."
                return resolved_command

            available_devices = self.comm_service.get_device_friendly_names()
            if not available_devices:
                resolved_command["response_on_failure"] = "Я не вижу доступных устройств для управления."
                return resolved_command

            search_text_for_matching = None
            if is_pronoun and last_mentioned_device_context:
                search_text_for_matching = last_mentioned_device_context
                logger.debug(f"Offline Processor: Pronoun used. Using context '{search_text_for_matching}' for device search.")
            elif device_description_from_nlu:
                search_text_for_matching = device_description_from_nlu
                logger.debug(f"Offline Processor: Using NLU device description '{search_text_for_matching}'.")
            elif intent in ['set_attribute', 'activate_device', 'deactivate_device'] and last_mentioned_device_context:
                # Fallback: Device control intent, no specific device in NLU, but context exists.
                search_text_for_matching = last_mentioned_device_context
                logger.debug(f"Offline Processor: No device in NLU for device intent. Using context '{search_text_for_matching}'.")
            
            if search_text_for_matching:
                normalized_search_text = _normalize_russian_text_with_pymorphy(search_text_for_matching)
                if normalized_search_text:
                    # Use a slightly higher cutoff for offline processing to be more certain
                    matched_device_info = find_best_match(normalized_search_text, available_devices, score_cutoff=75, return_match_info=True)
                    if matched_device_info:
                        resolved_device_name = matched_device_info["match"]
                        logger.info(f"Offline Processor: Device resolved via search '{search_text_for_matching}' (norm: '{normalized_search_text}') to '{resolved_device_name}' with score {matched_device_info['score']}.")
                        # IMPORTANT: Update context field as soon as device is resolved
                        resolved_command["resolved_device_name_for_context_update"] = resolved_device_name
                else:
                    logger.warning(f"Offline Processor: Search text '{search_text_for_matching}' became empty after normalization.")
            
            if not resolved_device_name:
                original_search_ref = device_description_from_nlu or \
                                      (last_mentioned_device_context if is_pronoun or not device_description_from_nlu else "не указанное устройство")
                resolved_command["response_on_failure"] = f"Я не уверена, какое устройство вы имеете в виду, говоря '{original_search_ref}'."
                # Do not clear resolved_device_name_for_context_update if it was set by a previous, more specific match attempt
                # that failed a later validation. However, in this path, it means no device was found at all.
                return resolved_command

            # Attribute and Value are already processed and standardized by RasaNLUEngine
            attribute_name = nlu_result.get("attribute") 
            raw_value_from_nlu = nlu_result.get("raw_value")

            if not attribute_name:
                resolved_command["response_on_failure"] = f"Я поняла, что речь о устройстве '{resolved_device_name}', но не поняла, что именно нужно изменить."
                return resolved_command
            if raw_value_from_nlu is None and intent == "set_attribute":
                resolved_command["response_on_failure"] = f"Для устройства '{resolved_device_name}', мне нужно знать, какое значение установить для '{attribute_name}'."
                return resolved_command

            resolved_command.update({
                "executable": True,
                "device_friendly_name": resolved_device_name,
                "attribute_name": attribute_name,
                "value_str": str(raw_value_from_nlu), # Tool expects string
                "tool_to_call": "set_device_attribute_directly",
                "tool_args": {
                    "device_friendly_name": resolved_device_name,
                    "attribute_name": attribute_name,
                    "value_str": str(raw_value_from_nlu)
                }
                # resolved_device_name_for_context_update is already set if device was found
            })
            return resolved_command
        else:
            resolved_command["response_on_failure"] = f"Я пока не умею обрабатывать команду '{intent}' в оффлайн режиме."
            return resolved_command

    async def execute_resolved_command(self, resolved_command: Dict[str, Any]) -> str:
        if not resolved_command.get("executable"):
            return resolved_command.get("response_on_failure", "Команду не удалось выполнить.")

        tool_to_call = resolved_command.get("tool_to_call")
        tool_args = resolved_command.get("tool_args", {})
        
        logger.info(f"Offline Processor: Executing '{tool_to_call}' with args: {tool_args}")

        try:
            if tool_to_call == "get_current_time_tool_invoke":
                # Ensure get_current_time.invoke can handle if timezone_str is None
                return get_current_time.invoke(tool_args) 
            elif tool_to_call == "set_device_attribute_directly":
                if not self.comm_service:
                     logger.error("Offline Processor: CommunicationService not available for set_device_attribute_directly.")
                     return "Ошибка: Сервис управления устройствами не доступен."
                return await set_device_attribute_directly(
                    comm_service=self.comm_service, # Pass the instance
                    **tool_args
                )
            else:
                logger.warning(f"Offline Processor: Unknown tool_to_call '{tool_to_call}'")
                return f"Простите, я не могу выполнить это действие: {tool_to_call}."
        except Exception as e:
            logger.error(f"Offline Processor: Error executing {tool_to_call}: {e}", exc_info=True)
            return f"Простите, произошла ошибка при попытке выполнить действие."