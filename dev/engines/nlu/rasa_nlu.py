import logging
import json
import httpx
from typing import Optional, Dict, Any

from .base import NLUEngineBase
from settings import RasaNLUSettings # Specific settings
from utils.text_processing import _normalize_russian_text_with_pymorphy, \
                                  _PRE_NORMALIZED_CITY_TO_TIMEZONE, _NLU_ATTR_TO_STANDARD, \
                                  initialize_city_timezones, morph, COLOR_NAME_TO_XY, \
                                  COLOR_TEMP_PRESETS, normalize_text_key
import re # For city name matching

logger = logging.getLogger(__name__)

# Ensure Pymorphy and city timezones are initialized when this module is loaded
if morph is None: 
    try:
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()
        logger.info("Pymorphy3 MorphAnalyzer initialized by RasaNLUEngine.")
    except Exception as e:
        logger.error(f"Failed to initialize Pymorphy3 in RasaNLUEngine: {e}")
        morph = None

initialize_city_timezones()

# REMOVED: _last_mentioned_device_context: Dict[str, Optional[str]] = {"value": None}

class RasaNLUEngine(NLUEngineBase):
    def __init__(self, config: Dict[str, Any]): # config is settings.rasa_nlu
        self.settings: RasaNLUSettings = config
        if not self.settings.url:
            logger.error("Rasa NLU URL not configured. RasaNLUEngine will not function.")

    async def parse(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.settings.url:
            logger.warning("RasaNLUEngine.parse called but URL is not configured.")
            return None

        payload = {"text": text}
        # Initialize with all expected keys to ensure consistent structure
        parsed_command_data: Dict[str, Any] = {
            "intent": None,
            "device_description_text": None, # Text describing the device
            "device_is_pronoun": False,      # Flag if a pronoun was used for the device
            "attribute": None,               # Standardized attribute name
            "raw_value": None,               # Normalized value for the attribute
            "timezone_str_for_tool": None,   # For get_time intent
            "original_text": text,           # Store original text for context/debugging
            "entities_raw": []               # Store raw entities from NLU
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
            parsed_command_data["entities_raw"] = entities # Store raw entities
            
            if not intent_name or confidence < self.settings.intent_confidence_threshold:
                logger.info(f"NLU confidence low ({confidence:.2f} < {self.settings.intent_confidence_threshold}) for intent '{intent_name}' or intent not found. Text: '{text}'.")
                return None # Or return parsed_command_data with intent=None if caller handles it
            
            parsed_command_data["intent"] = intent_name
            logger.info(f"Rasa NLU classified intent as '{intent_name}' with confidence {confidence:.2f}")

            # --- Intent-Specific Parsing ---
            if intent_name == "get_time":
                timezone_entity = next((e for e in entities if e['entity'] == 'timezone_location'), None)
                resolved_timezone_arg = None

                if timezone_entity:
                    user_specified_location = timezone_entity['value']
                    normalized_location_text = _normalize_russian_text_with_pymorphy(user_specified_location)
                    
                    mapped_tz_db_name = None
                    # Sort by length to match longer city names first (e.g., "Нижний Новгород" before "Новгород")
                    sorted_city_keys = sorted(_PRE_NORMALIZED_CITY_TO_TIMEZONE.keys(), key=len, reverse=True)

                    for known_norm_city in sorted_city_keys:
                        # Use word boundaries for more precise matching
                        if re.search(r'\b' + re.escape(known_norm_city) + r'\b', normalized_location_text, re.IGNORECASE):
                            mapped_tz_db_name = _PRE_NORMALIZED_CITY_TO_TIMEZONE[known_norm_city]
                            logger.debug(f"Timezone: Matched '{user_specified_location}' (norm: '{normalized_location_text}') to '{known_norm_city}' -> '{mapped_tz_db_name}'")
                            break
                    
                    if not mapped_tz_db_name and normalized_location_text in _PRE_NORMALIZED_CITY_TO_TIMEZONE:
                         # Fallback for exact match on normalized text if regex didn't catch it (should be rare)
                         mapped_tz_db_name = _PRE_NORMALIZED_CITY_TO_TIMEZONE[normalized_location_text]
                         logger.debug(f"Timezone: Matched '{user_specified_location}' (norm: '{normalized_location_text}') directly to '{mapped_tz_db_name}'")


                    if mapped_tz_db_name:
                        resolved_timezone_arg = mapped_tz_db_name
                    else:
                        logger.warning(f"Could not map NLU timezone location '{user_specified_location}' (norm: '{normalized_location_text}') to a known TZ. Passing raw.")
                        resolved_timezone_arg = user_specified_location # Pass raw if no match
                
                parsed_command_data["timezone_str_for_tool"] = resolved_timezone_arg
                return parsed_command_data

            elif intent_name in ["activate_device", "deactivate_device", "set_attribute"]:
                device_description_parts = [e['value'] for e in entities if e['entity'] == 'device_description']
                location_parts = [e['value'] for e in entities if e['entity'] == 'location']
                
                # Combine device description and location parts
                raw_device_search_parts = device_description_parts + location_parts
                # Filter out None or empty strings before joining
                device_desc_text = " ".join(filter(None, raw_device_search_parts)).strip()
                if device_desc_text: # Only set if there's actual text
                    parsed_command_data["device_description_text"] = device_desc_text
                
                pronoun_entity = next((e for e in entities if e['entity'] == 'pronoun_device'), None)
                if pronoun_entity:
                    parsed_command_data["device_is_pronoun"] = True
                    logger.debug("Pronoun detected for device.")

                attribute_name_entity = next((e for e in entities if e['entity'] == 'attribute_name'), None)
                if attribute_name_entity:
                    nlu_attr_val = attribute_name_entity['value']
                    # Normalize the attribute name extracted from NLU
                    norm_nlu_attr_val = normalize_text_key(_normalize_russian_text_with_pymorphy(nlu_attr_val))
                    parsed_command_data["attribute"] = _NLU_ATTR_TO_STANDARD.get(norm_nlu_attr_val, norm_nlu_attr_val)
                    logger.debug(f"Attribute extracted: '{nlu_attr_val}', normalized to: '{parsed_command_data['attribute']}'")


                # Extract attribute value, considering multiple entities might form the value
                attribute_value_entities = sorted(
                    [e for e in entities if e['entity'] == 'attribute_value'],
                    key=lambda e: e['start'] # Sort by start position to reconstruct multi-word values
                )
                attribute_value_raw = " ".join(e['value'] for e in attribute_value_entities).strip() if attribute_value_entities else None
                
                # Fallback: if no 'attribute_value' entity, check for standalone 'percentage' or 'number'
                if not attribute_value_raw:
                    numeric_entity = next((e for e in entities if e['entity'] in ['percentage', 'number']), None)
                    if numeric_entity:
                         attribute_value_raw = numeric_entity['value']
                         logger.debug(f"Attribute value taken from numeric entity: '{attribute_value_raw}'")

                
                if attribute_value_raw:
                     # Normalize the raw value text
                     parsed_command_data["raw_value"] = _normalize_russian_text_with_pymorphy(attribute_value_raw)
                     logger.debug(f"Raw value extracted: '{attribute_value_raw}', normalized to: '{parsed_command_data['raw_value']}'")


                # Guess attribute if not explicitly found but a value is present
                if not parsed_command_data.get("attribute") and attribute_value_raw:
                    # Use the first entity that contributed to attribute_value_raw, or a numeric entity
                    primary_attribute_value_entity = attribute_value_entities[0] if attribute_value_entities else \
                                                    next((e for e in entities if e['entity'] in ['percentage', 'number']), None)
                    if primary_attribute_value_entity:
                        entity_group = primary_attribute_value_entity.get('group') # Rasa entity groups
                        guessed_attr = None
                        if entity_group == 'color_values': guessed_attr = "color"
                        elif entity_group == 'temp_presets': guessed_attr = "color_temp"
                        elif entity_group == 'relative_brightness_values': guessed_attr = "brightness"
                        elif primary_attribute_value_entity.get('entity') == 'percentage': guessed_attr = "brightness"
                        else: # Try to guess based on normalized value content
                            normalized_val_for_guess = normalize_text_key(_normalize_russian_text_with_pymorphy(attribute_value_raw))
                            if normalized_val_for_guess in COLOR_NAME_TO_XY: guessed_attr = "color"
                            elif normalized_val_for_guess in COLOR_TEMP_PRESETS: guessed_attr = "color_temp"
                        
                        if guessed_attr:
                            parsed_command_data["attribute"] = guessed_attr
                            logger.info(f"Guessed attribute '{guessed_attr}' from value '{attribute_value_raw}'.")
                
                # Standardize attribute and value for activate/deactivate intents
                if intent_name == "activate_device":
                    parsed_command_data["attribute"] = "state" # Standard attribute for on/off
                    parsed_command_data["raw_value"] = "ON"    # Standard value
                elif intent_name == "deactivate_device":
                    parsed_command_data["attribute"] = "state"
                    parsed_command_data["raw_value"] = "OFF"

                # Basic validation for set_attribute: requires an attribute and a value.
                if intent_name == "set_attribute" and \
                   (not parsed_command_data.get("attribute") or parsed_command_data.get("raw_value") is None):
                    logger.warning(f"Incomplete 'set_attribute' intent from NLU: missing attribute or value. Data: {parsed_command_data}")
                    # Return None or the partial data? Returning None signals failure to parse completely.
                    return None 
                
                return parsed_command_data
            else:
                logger.warning(f"Unhandled Rasa intent '{intent_name}'. No specific parsing logic implemented.")
                # Return the data with just the intent if the caller wants to handle unknown intents.
                # Or return None if only fully understood intents are processed.
                return parsed_command_data # Or None

        except httpx.RequestError as e:
            logger.error(f"Rasa NLU connection error to {self.settings.url}: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Rasa NLU server error {e.response.status_code}. Response: {e.response.text[:200]}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from Rasa NLU: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
        except Exception as e:
            logger.error(f"Unexpected error during Rasa NLU parsing: {e}", exc_info=True)
        
        return None # Return None on any error during HTTP request or parsing