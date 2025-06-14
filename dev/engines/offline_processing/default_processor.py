# engines/offline_processing/default_processor.py

import logging
import json
import re # Added import
from typing import Dict, Any, Optional

from .base import OfflineCommandProcessorBase
from engines.communication.base import CommunicationServiceBase

from utils.text_processing import find_best_match, _normalize_russian_text_with_pymorphy, \
                                  _PRE_NORMALIZED_CITY_TO_TIMEZONE, _NLU_ATTR_TO_STANDARD, \
                                  initialize_city_timezones, morph, COLOR_NAME_TO_XY, \
                                  COLOR_TEMP_PRESETS, normalize_text_key

from tools.time import get_current_time
from tools.device_control import set_device_attribute_directly

logger = logging.getLogger(__name__)

# Initialization logic remains the same
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
                                 ) -> Dict[str, Any]:
        """
        Processes raw NLU output, resolves entities, uses context, and builds an executable command.
        This is the central hub for offline command logic.
        """
        intent = nlu_result.get("intent")
        entities = nlu_result.get("entities", [])
        
        resolved_command = {
            "executable": False,
            "intent": intent,
            "nlu_raw_output": nlu_result,
            "response_on_failure": "Простите, я не смогла полностью понять или обработать вашу команду в оффлайн режиме."
        }

        if not intent:
            resolved_command["response_on_failure"] = "Простите, я не поняла, что вы хотите сделать."
            return resolved_command

        # --- GET_TIME INTENT ---
        if intent == "get_time":
            # All parsing logic is now centralized here.
            timezone_entity = next((e for e in entities if e['entity'] == 'timezone_location'), None)
            resolved_timezone_arg = None

            if timezone_entity:
                user_specified_location = timezone_entity['value']
                normalized_location_text = _normalize_russian_text_with_pymorphy(user_specified_location)
                
                mapped_tz_db_name = None
                sorted_city_keys = sorted(_PRE_NORMALIZED_CITY_TO_TIMEZONE.keys(), key=len, reverse=True)
                for known_norm_city in sorted_city_keys:
                    tz_db_name = _PRE_NORMALIZED_CITY_TO_TIMEZONE[known_norm_city]
                    if re.search(r'\b' + re.escape(known_norm_city) + r'\b', normalized_location_text, re.IGNORECASE):
                        mapped_tz_db_name = tz_db_name
                        break
                if not mapped_tz_db_name and normalized_location_text in _PRE_NORMALIZED_CITY_TO_TIMEZONE:
                        mapped_tz_db_name = _PRE_NORMALIZED_CITY_TO_TIMEZONE[normalized_location_text]

                if mapped_tz_db_name:
                    resolved_timezone_arg = mapped_tz_db_name
                else:
                    logger.warning(f"Could not map NLU timezone location '{user_specified_location}' to TZ. Passing raw.")
                    resolved_timezone_arg = user_specified_location
            
            resolved_command.update({
                "executable": True,
                "tool_to_call": "get_current_time_tool_invoke",
                "tool_args": {"timezone_str": resolved_timezone_arg}
            })
            return resolved_command

        # --- DEVICE CONTROL INTENTS ---
        elif intent in ["activate_device", "deactivate_device", "set_attribute"]:
            # == Step 1: Parse all entities from raw NLU output ==
            device_description_parts = [e['value'] for e in entities if e['entity'] == 'device_description']
            location_parts = [e['value'] for e in entities if e['entity'] == 'location']
            device_description_from_nlu = " ".join(filter(None, device_description_parts + location_parts)).strip()
            
            is_pronoun = any(e['entity'] == 'pronoun_device' for e in entities)

            attribute_name_entity = next((e for e in entities if e['entity'] == 'attribute_name'), None)
            attribute_name = None
            if attribute_name_entity:
                nlu_attr_val = attribute_name_entity['value']
                norm_nlu_attr_val = normalize_text_key(_normalize_russian_text_with_pymorphy(nlu_attr_val))
                attribute_name = _NLU_ATTR_TO_STANDARD.get(norm_nlu_attr_val, norm_nlu_attr_val)

            attribute_value_entities = sorted([e for e in entities if e['entity'] == 'attribute_value'], key=lambda e: e['start'])
            raw_value_from_nlu = " ".join(e['value'] for e in attribute_value_entities).strip() if attribute_value_entities else None
            
            if not raw_value_from_nlu:
                numeric_entity = next((e for e in entities if e['entity'] in ['percentage', 'number']), None)
                if numeric_entity:
                    raw_value_from_nlu = numeric_entity['value']
            
            if raw_value_from_nlu:
                raw_value_from_nlu = _normalize_russian_text_with_pymorphy(raw_value_from_nlu)

            # Guess attribute if not explicitly found
            if not attribute_name and raw_value_from_nlu:
                primary_value_entity = attribute_value_entities[0] if attribute_value_entities else \
                                       next((e for e in entities if e['entity'] in ['percentage', 'number']), None)
                if primary_value_entity:
                    entity_group = primary_value_entity.get('group')
                    if entity_group == 'color_values': attribute_name = "color"
                    elif entity_group == 'temp_presets': attribute_name = "color_temp"
                    elif entity_group == 'relative_brightness_values': attribute_name = "brightness"
                    elif primary_value_entity.get('entity') == 'percentage': attribute_name = "brightness"
                    else:
                        normalized_val = normalize_text_key(raw_value_from_nlu)
                        if normalized_val in COLOR_NAME_TO_XY: attribute_name = "color"
                        elif normalized_val in COLOR_TEMP_PRESETS: attribute_name = "color_temp"
                    if attribute_name:
                        logger.info(f"Guessed attribute '{attribute_name}' from value '{raw_value_from_nlu}'.")

            # Set value for activate/deactivate
            if intent == "activate_device":
                attribute_name = "state"
                raw_value_from_nlu = "ON"
            elif intent == "deactivate_device":
                attribute_name = "state"
                raw_value_from_nlu = "OFF"

            # == Step 2: Resolve device name using context and parsed entities ==
            resolved_device_name: Optional[str] = None
            available_devices = self.comm_service.get_device_friendly_names()

            if not available_devices:
                resolved_command["response_on_failure"] = "Я не вижу доступных устройств для управления."
                return resolved_command

            search_text_for_matching = device_description_from_nlu
            if is_pronoun and last_mentioned_device_context:
                search_text_for_matching = last_mentioned_device_context
                logger.debug(f"Offline Processor: Pronoun indicated. Using context '{search_text_for_matching}' for device search.")
            
            if search_text_for_matching:
                normalized_search_text = _normalize_russian_text_with_pymorphy(search_text_for_matching)
                if normalized_search_text:
                    matched_device = find_best_match(normalized_search_text, available_devices, score_cutoff=70)
                    if matched_device:
                        resolved_device_name = matched_device
                        logger.info(f"Offline Processor: Device resolved via search '{search_text_for_matching}' (norm: '{normalized_search_text}') to '{resolved_device_name}'")
                else:
                    logger.warning(f"Offline Processor: Search text '{search_text_for_matching}' became empty after normalization.")
            
            if not resolved_device_name:
                original_search_ref = device_description_from_nlu or (last_mentioned_device_context if is_pronoun else "не указанное устройство")
                resolved_command["response_on_failure"] = f"Я не уверена, какое устройство вы имеете в виду, говоря '{original_search_ref}'."
                return resolved_command

            # == Step 3: Validate and build the final executable command ==
            if not attribute_name:
                resolved_command["response_on_failure"] = f"Я поняла, что речь о устройстве '{resolved_device_name}', но не поняла, что именно нужно изменить."
                return resolved_command
            if raw_value_from_nlu is None and intent == "set_attribute":
                resolved_command["response_on_failure"] = f"Для устройства '{resolved_device_name}', мне нужно знать, какое значение установить для '{attribute_name}'."
                return resolved_command

            resolved_command.update({
                "executable": True,
                "tool_to_call": "set_device_attribute_directly",
                "tool_args": {
                    "device_friendly_name": resolved_device_name,
                    "attribute_name": attribute_name,
                    "value_str": str(raw_value_from_nlu)
                },
                "resolved_device_name_for_context_update": resolved_device_name
            })
            return resolved_command
        else:
            resolved_command["response_on_failure"] = f"Я пока не умею обрабатывать команду '{intent}' в оффлайн режиме."
            return resolved_command

    async def execute_resolved_command(self, resolved_command: Dict[str, Any]) -> str:
        # This method does not need to change.
        if not resolved_command.get("executable"):
            return resolved_command.get("response_on_failure", "Команду не удалось выполнить.")

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
                return f"Простите, я не могу выполнить это действие: {tool_to_call}."
        except Exception as e:
            logger.error(f"Offline Processor: Error executing {tool_to_call}: {e}", exc_info=True)
            return f"Простите, произошла ошибка при попытке выполнить действие."