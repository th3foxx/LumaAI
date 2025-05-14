import logging
import json
import httpx
from typing import Optional, Dict, Any

from .base import NLUEngineBase
from settings import RasaNLUSettings # Specific settings
# Import helpers from the original offline_controller or move them here/to a utils module
from utils.text_processing import _normalize_russian_text_with_pymorphy, find_best_match, \
                                  _PRE_NORMALIZED_CITY_TO_TIMEZONE, _NLU_ATTR_TO_STANDARD, \
                                  initialize_city_timezones, morph, COLOR_NAME_TO_XY, \
                                  COLOR_TEMP_PRESETS, normalize_text_key # (NEW)
import re # For city name matching

logger = logging.getLogger(__name__)

# Ensure Pymorphy and city timezones are initialized when this module is loaded
if morph is None: # Attempt to initialize if not already done by offline_controller's import
    try:
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()
        logger.info("Pymorphy3 MorphAnalyzer initialized by RasaNLUEngine.")
    except Exception as e:
        logger.error(f"Failed to initialize Pymorphy3 in RasaNLUEngine: {e}")
        morph = None # Explicitly set to None

initialize_city_timezones() # Re-run to ensure it uses the potentially newly initialized morph

_last_mentioned_device_context: Dict[str, Optional[str]] = {"value": None} # Simple context store

class RasaNLUEngine(NLUEngineBase):
    def __init__(self, config: Dict[str, Any]): # config is settings.rasa_nlu
        self.settings: RasaNLUSettings = config
        if not self.settings.url:
            logger.error("Rasa NLU URL not configured. RasaNLUEngine will not function.")

    async def parse(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.settings.url:
            return None

        payload = {"text": text}
        parsed_command_data: Dict[str, Any] = {
            "intent": None, "device": None, "attribute": None,
            "raw_value": None, "timezone_str_for_tool": None,
        }
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
            entities = result.get("entities", [])
            
            parsed_command_data["intent"] = intent_name

            if not intent_name or confidence < self.settings.intent_confidence_threshold:
                logger.warning(f"NLU confidence low ({confidence:.2f} < {self.settings.intent_confidence_threshold}) for intent '{intent_name}'. Text: '{text}'.")
                return None
            logger.info(f"Rasa NLU classified intent as '{intent_name}' with confidence {confidence:.2f}")

            # --- Intent-Specific Parsing (adapted from original offline_controller.py) ---
            if intent_name == "get_time":
                # ... (Copy and adapt get_time parsing logic from offline_controller.parse_offline_command)
                # Important: Replace global _last_mentioned_device with a context passed in or managed by ConnectionManager
                # For now, using a module-level context for simplicity in this direct adaptation.
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
                
                parsed_command_data["timezone_str_for_tool"] = resolved_timezone_arg
                return parsed_command_data

            elif intent_name in ["activate_device", "deactivate_device", "set_attribute"]:
                # ... (Copy and adapt device control parsing logic from offline_controller.parse_offline_command)
                # This needs access to MQTT device list. The NLU engine itself shouldn't directly call MQTT.
                # This implies that device resolution might need to happen one layer up, or the NLU engine
                # needs to be provided with the list of available devices during parsing.
                # For now, let's assume it can't resolve against live MQTT devices here.
                # It should extract what it can, and the caller (ConnectionManager) can do final resolution.
                # Or, we pass `mqtt_client.get_device_friendly_names()` to this parse method.
                # Let's assume for now it just extracts entities, and resolution happens later.
                # However, the original code does resolution here.
                # To keep it similar: the CommunicationServiceBase needs to be accessible.
                # This is tricky. For a pure NLU, it shouldn't know about MQTT.
                # Let's simplify: NLU extracts device description, ConnectionManager resolves.
                # For this refactor, I'll keep the original logic flow which means NLU needs device names.
                # This is a design compromise to limit scope of changes from original.
                # It implies the NLU engine might need a reference to the communication service,
                # or the list of devices is passed into the parse method.
                # Let's assume `config` can provide `get_available_devices_func`.
                # This is getting complicated. The original `parse_offline_command` used a global mqtt_client.

                # Simpler approach for now: extract entities, let ConnectionManager handle resolution.
                # This means the output of this NLU will be slightly different than the original `parse_offline_command`.
                # It will return `device_description_text` instead of resolved `device`.

                device_description_parts = [e['value'] for e in entities if e['entity'] == 'device_description']
                location_parts = [e['value'] for e in entities if e['entity'] == 'location']
                raw_device_search_parts = device_description_parts + location_parts
                parsed_command_data["device_description_text"] = " ".join(filter(None, raw_device_search_parts)).strip()
                
                pronoun_entity = next((e for e in entities if e['entity'] == 'pronoun_device'), None)
                if pronoun_entity:
                    parsed_command_data["device_is_pronoun"] = True


                attribute_name_entity = next((e for e in entities if e['entity'] == 'attribute_name'), None)
                if attribute_name_entity:
                    nlu_attr_val = attribute_name_entity['value']
                    norm_nlu_attr_val = normalize_text_key(_normalize_russian_text_with_pymorphy(nlu_attr_val))
                    parsed_command_data["attribute"] = _NLU_ATTR_TO_STANDARD.get(norm_nlu_attr_val, norm_nlu_attr_val)

                attribute_value_entities = [e for e in entities if e['entity'] == 'attribute_value']
                raw_attribute_values = sorted(attribute_value_entities, key=lambda e: e['start'])
                attribute_value_raw = " ".join(e['value'] for e in raw_attribute_values).strip() if raw_attribute_values else None
                
                if not attribute_value_raw: # Check standalone number/percentage
                    numeric_entity = next((e for e in entities if e['entity'] in ['percentage', 'number']), None)
                    if numeric_entity:
                         attribute_value_raw = numeric_entity['value']
                
                if attribute_value_raw:
                     parsed_command_data["raw_value"] = _normalize_russian_text_with_pymorphy(attribute_value_raw)


                # Guess attribute if not explicitly found
                if not parsed_command_data.get("attribute") and attribute_value_raw:
                    primary_attribute_value_entity = raw_attribute_values[0] if raw_attribute_values else \
                                                    next((e for e in entities if e['entity'] in ['percentage', 'number']), None)
                    if primary_attribute_value_entity:
                        entity_group = primary_attribute_value_entity.get('group')
                        guessed_attr = None
                        if entity_group == 'color_values': guessed_attr = "color"
                        elif entity_group == 'temp_presets': guessed_attr = "color_temp"
                        elif entity_group == 'relative_brightness_values': guessed_attr = "brightness"
                        elif primary_attribute_value_entity.get('entity') == 'percentage': guessed_attr = "brightness"
                        else:
                            normalized_val = normalize_text_key(_normalize_russian_text_with_pymorphy(attribute_value_raw))
                            if normalized_val in COLOR_NAME_TO_XY: guessed_attr = "color"
                            elif normalized_val in COLOR_TEMP_PRESETS: guessed_attr = "color_temp"
                        if guessed_attr:
                            parsed_command_data["attribute"] = guessed_attr
                            logger.info(f"Guessed attribute '{guessed_attr}' from value.")
                
                # Set value for activate/deactivate
                if intent_name == "activate_device":
                    parsed_command_data["attribute"] = "state"
                    parsed_command_data["raw_value"] = "ON"
                elif intent_name == "deactivate_device":
                    parsed_command_data["attribute"] = "state"
                    parsed_command_data["raw_value"] = "OFF"

                # Store last mentioned device description for pronoun resolution by ConnectionManager
                if parsed_command_data.get("device_description_text"):
                    _last_mentioned_device_context["value"] = parsed_command_data["device_description_text"]
                elif parsed_command_data.get("device_is_pronoun") and _last_mentioned_device_context["value"]:
                    # If it's a pronoun, fill description from context for the caller
                    parsed_command_data["device_description_text"] = _last_mentioned_device_context["value"]


                # Basic validation for set_attribute
                if intent_name == "set_attribute" and \
                   (not parsed_command_data.get("attribute") or parsed_command_data.get("raw_value") is None):
                    logger.warning(f"Incomplete 'set_attribute': {parsed_command_data}")
                    return None
                
                return parsed_command_data
            else:
                logger.warning(f"Unhandled Rasa intent '{intent_name}'.")
                return None

        except httpx.RequestError as e:
            logger.error(f"Rasa NLU connection error to {self.settings.url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Rasa NLU server error {e.response.status_code}. Response: {e.response.text[:200]}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from Rasa NLU: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Rasa NLU parsing: {e}", exc_info=True)
        return None