import asyncio
import json
import logging
from typing import Dict, Optional, Any, List

from langchain_core.tools import tool # Keep for decorating the LLM-facing tool
from engines.communication.base import CommunicationServiceBase
from settings import settings

from .color_presets import COLOR_NAME_TO_XY, COLOR_TEMP_PRESETS, normalize_text_key
from thefuzz import process as fuzz_process, fuzz

logger = logging.getLogger(__name__)

TOOL_COMM_SERVICE: Optional[CommunicationServiceBase] = None

def initialize_device_control_tool(comm_service_instance: CommunicationServiceBase):
    global TOOL_COMM_SERVICE
    TOOL_COMM_SERVICE = comm_service_instance
    logger.info("Device control tool initialized with communication service.")

def normalize_attribute_name(attr_name: str) -> str:
    return attr_name.lower().replace(" ", "_")

def parse_value(value_str: str, target_type: str) -> Optional[Any]:
    # ... (implementation as before) ...
    try:
        if target_type == "numeric":
             try: return int(value_str)
             except ValueError: return float(value_str)
        elif target_type == "binary":
            val_lower = value_str.strip().lower()
            if val_lower in ["on", "вкл", "включить", "включи", "1", "true", "да"]: return "ON"
            if val_lower in ["off", "выкл", "выключить", "выключи", "0", "false", "нет"]: return "OFF"
            if val_lower in ["toggle", "переключить", "переключи"]: return "TOGGLE"
            return None
        elif target_type == "enum":
            return value_str 
        else: 
            return value_str
    except (ValueError, TypeError):
        logger.warning(f"Could not parse value '{value_str}' as {target_type}")
        return None


def find_capability(capabilities: Dict[str, Dict[str, Any]], requested_attr: str) -> Optional[Dict[str, Any]]:
    # ... (implementation as before) ...
    if not capabilities: return None
    normalized_requested = normalize_attribute_name(requested_attr)
    for cap_name, details in capabilities.items():
        if normalize_attribute_name(cap_name) == normalized_requested:
            return details
        if normalize_attribute_name(details.get("name", "")) == normalized_requested:
            return details
    logger.debug(f"Capability '{requested_attr}' (normalized: {normalized_requested}) not found directly.")
    return None


async def _set_device_attribute_core_logic( # NEW ASYNC HELPER FUNCTION
    comm_service: CommunicationServiceBase, # Explicitly takes comm_service
    user_description: str,
    attribute: str,
    value: str,
    guessed_friendly_name: Optional[str] = None
) -> str:
    """
    Core logic for setting a device attribute.
    This function is NOT decorated with @tool.
    """
    if comm_service is None: # Check the passed instance
        logger.error("Core device control logic: communication service is None.")
        return "Error: Device control service is not configured internally."
    
    if not comm_service.is_connected:
        return "Error: Not connected to the device control system."

    if not user_description or not attribute or value is None:
        return "Error: Missing required parameters for device control."

    available_devices = comm_service.get_device_friendly_names()
    if not available_devices:
        return "Sorry, the device list is unavailable from the control system."

    # --- 1. Find target device ---
    target_device_name = None
    # If guessed_friendly_name is provided and it's a good direct match, use it.
    # This is useful when set_device_attribute_directly calls this,
    # as device_friendly_name is already resolved.
    if guessed_friendly_name and guessed_friendly_name in available_devices:
        # A simple direct check first if guessed_friendly_name is an exact match
        # This is especially true if called from `set_device_attribute_directly`
        # where `guessed_friendly_name` IS the `device_friendly_name`.
        is_exact_match = False
        for dev_name in available_devices:
            if normalize_attribute_name(guessed_friendly_name) == normalize_attribute_name(dev_name):
                target_device_name = dev_name
                is_exact_match = True
                logger.info(f"Target device directly set by 'guessed_friendly_name': '{target_device_name}'")
                break
        if not is_exact_match and guessed_friendly_name: # Fallback to fuzzy if not exact
             # Validate guessed_friendly_name with a high threshold if it wasn't an exact match
            best_guess_match = fuzz_process.extractOne(guessed_friendly_name, available_devices, scorer=fuzz.token_sort_ratio)
            if best_guess_match and best_guess_match[1] >= 85: # Higher threshold for guess
                target_device_name = best_guess_match[0]
                logger.info(f"Target device confirmed by 'guessed_friendly_name' (fuzzy): '{target_device_name}' score {best_guess_match[1]}")


    if not target_device_name and user_description: # If guess didn't work or wasn't provided, use user_description
        description_match_threshold = 70 # Keep this threshold reasonable
        best_desc_match = fuzz_process.extractOne(user_description, available_devices, scorer=fuzz.token_sort_ratio)
        if best_desc_match and best_desc_match[1] >= description_match_threshold:
            target_device_name = best_desc_match[0]
            logger.info(f"Target device by user_description '{user_description}': '{target_device_name}' score {best_desc_match[1]}")
        else:
            err_msg = f"Sorry, I couldn't find a device matching '{user_description}'."
            if best_desc_match: err_msg += f" Closest was '{best_desc_match[0]}' (score: {best_desc_match[1]})."
            return err_msg + f" Available: {available_devices[:5]}"

    if not target_device_name: # Should not happen if logic above is correct
        return "Sorry, could not identify the target device."

    logger.info(f"Core Logic: Target device identified as '{target_device_name}'")

    # --- 2. Get capabilities ---
    capabilities = comm_service.get_device_capabilities(target_device_name)
    if capabilities is None:
        return f"Sorry, could not get capabilities for '{target_device_name}'."

    # --- 3. Find capability ---
    capability_details = find_capability(capabilities, attribute)
    if not capability_details:
        # ... (error message as before) ...
        available_attrs = list(capabilities.keys())
        fuzzy_attr_match = fuzz_process.extractOne(attribute, available_attrs, scorer=fuzz.token_sort_ratio, score_cutoff=80)
        suggestion = f" Did you mean '{fuzzy_attr_match[0]}'?" if fuzzy_attr_match else ""
        return f"Sorry, '{target_device_name}' doesn't support '{attribute}'.{suggestion} Available: {available_attrs[:5]}"


    # --- 4. Check access ---
    if not (capability_details.get("access", 0) & 2):
        return f"Sorry, '{capability_details.get('name', attribute)}' on '{target_device_name}' is read-only."

    # --- 5. Transform, validate value, create payload ---
    # (This extensive logic for color, color_temp, numeric, binary, enum remains the same)
    # Ensure it uses `comm_service` where needed (though it mostly uses capability_details)
    # and `settings` for topic base.
    cap_type = capability_details.get("type")
    cap_name_from_expose = capability_details.get('name', attribute)
    mqtt_attribute_name = capability_details.get('property', cap_name_from_expose)
    if not mqtt_attribute_name:
         return f"Internal error: Could not find MQTT property for '{attribute}' on '{target_device_name}'."

    payload_dict = None
    final_value_repr = value 

    if mqtt_attribute_name == 'color' or (cap_name_from_expose == 'color' and cap_type == 'light'):
        supports_xy = False
        if cap_type == 'light' and 'features' in capability_details:
            supports_xy = any(f.get('property') == 'color_xy' or f.get('name') == 'color_xy' for f in capability_details['features'])
        elif cap_type == 'composite' and mqtt_attribute_name == 'color':
            supports_xy = all(p in [f.get('property') for f in capability_details.get('features', [])] for p in ['x', 'y'])
        if not supports_xy:
             return f"Device '{target_device_name}' doesn't support XY color control."
        normalized_color = normalize_text_key(value)
        if normalized_color in COLOR_NAME_TO_XY:
            xy_coords = COLOR_NAME_TO_XY[normalized_color]
            payload_dict = {"color": {"x": xy_coords[0], "y": xy_coords[1]}}
            final_value_repr = f"{normalized_color} (xy: {xy_coords})"
        else:
            return f"Unrecognized color '{value}'. Try red, blue, etc."
    elif mqtt_attribute_name == 'color_temp' or (cap_name_from_expose == 'color_temp' and cap_type == 'light'):
        normalized_value_key = normalize_text_key(value)
        preset_mired_value = COLOR_TEMP_PRESETS.get(normalized_value_key)
        final_mired_value = None
        if preset_mired_value is not None:
            final_mired_value = preset_mired_value
            final_value_repr = f"{normalized_value_key} ({preset_mired_value} Mired)"
        else:
            if 'k' in value.lower(): return "Kelvin not supported. Use Mired or presets."
            parsed_as_numeric = parse_value(value, "numeric")
            if isinstance(parsed_as_numeric, (int, float)):
                final_mired_value = parsed_as_numeric; final_value_repr = str(final_mired_value)
            else: return f"'{value}' not a recognized color temp preset or Mired value."
        if final_mired_value is not None:
            min_val = capability_details.get("value_min"); max_val = capability_details.get("value_max")
            if (min_val is not None and final_mired_value < min_val) or \
               (max_val is not None and final_mired_value > max_val):
                return f"Value {final_mired_value} for color_temp out of range ({min_val}-{max_val})."
            payload_dict = {mqtt_attribute_name: int(round(final_mired_value))}
    else:
        parsed_value = parse_value(value, cap_type)
        final_value = parsed_value
        final_value_repr = str(final_value) if final_value is not None else value
        if final_value is None and cap_type != "enum":
             return f"Couldn't understand value '{value}' for '{cap_name_from_expose}' (type: {cap_type})."
        if cap_type == "numeric":
            if not isinstance(final_value, (int, float)): return f"Expected numeric for '{cap_name_from_expose}', got '{value}'."
            if (mqtt_attribute_name == 'brightness' or cap_name_from_expose == 'brightness') and isinstance(value, str) and '%' in value:
                 try:
                     percent = float(value.replace('%','').strip()); percent = max(0.0, min(100.0, percent))
                     abs_max = float(capability_details.get("value_max", 254.0)); abs_min = float(capability_details.get("value_min", 0.0))
                     calc_value = abs_min + (abs_max - abs_min) * (percent / 100.0)
                     final_value = int(round(calc_value)); final_value_repr = f"{percent}% ({final_value})"
                 except (ValueError, TypeError): return f"Couldn't parse percentage '{value}' for brightness."
            min_val = capability_details.get("value_min"); max_val = capability_details.get("value_max")
            if (min_val is not None and final_value < min_val) or \
               (max_val is not None and final_value > max_val):
                return f"Value {final_value} for '{cap_name_from_expose}' out of range ({min_val}-{max_val})."
            payload_dict = {mqtt_attribute_name: final_value}
        elif cap_type == "binary":
            allowed_vals = [capability_details.get("value_on", "ON"), capability_details.get("value_off", "OFF")]
            if capability_details.get("value_toggle"): allowed_vals.append(capability_details.get("value_toggle"))
            if final_value not in allowed_vals: return f"Invalid state '{value}'. Allowed: {allowed_vals}"
            payload_dict = {mqtt_attribute_name: final_value}
        elif cap_type == "enum":
            allowed_enum = capability_details.get("values")
            if not allowed_enum or final_value not in allowed_enum:
                suggestion = ""
                if allowed_enum:
                    enum_match = fuzz_process.extractOne(value, allowed_enum, scorer=fuzz.token_sort_ratio, score_cutoff=75)
                    if enum_match: suggestion = f" Did you mean '{enum_match[0]}'?"
                return f"Invalid option '{value}' for '{cap_name_from_expose}'.{suggestion} Allowed: {allowed_enum}"
            payload_dict = {mqtt_attribute_name: final_value}
        elif cap_type == "text" or cap_type == "string":
            payload_dict = {mqtt_attribute_name: final_value}
        else:
            return f"I don't know how to set attribute '{cap_name_from_expose}' of type '{cap_type}'."

    if payload_dict is None:
        return f"Internal error: Could not determine payload for '{attribute}' on '{target_device_name}' with value '{value}'."

    # --- 6. Send Command ---
    payload_json = json.dumps(payload_dict)
    topic = f"{settings.mqtt_broker.default_topic_base}/{target_device_name}/set"

    logger.info(f"Core Logic: Setting '{mqtt_attribute_name}' for '{target_device_name}'. Topic: {topic}, Payload: {payload_json}")
    
    success = await comm_service.publish(topic, payload_json, qos=1)

    if success:
        return f"Okay. '{target_device_name}' {cap_name_from_expose} set to {final_value_repr}."
    else:
        return f"Sorry, there was an issue sending the command for '{target_device_name}'."


@tool
async def set_device_attribute( # This is what the LLM will call
    user_description: str,
    attribute: str,
    value: str,
    guessed_friendly_name: Optional[str] = None
) -> str:
    """
    Sets a specific attribute of a Zigbee device based on its user-friendly description or a guessed name.
    (Docstring can be slightly simplified as the core logic details are now in the helper)
    Use this to control features like on/off state, brightness, color temperature, or color.
    Parameters:
        user_description (str): User's description (e.g., 'light in the hall', 'kitchen lamp').
        attribute (str): Attribute to change (e.g., 'state', 'brightness', 'color_temp', 'color').
        value (str): Desired value (e.g., 'ON', '75', 'cool', 'red').
        guessed_friendly_name (Optional[str]): LLM's best guess for the exact 'friendly_name'.
    """
    global TOOL_COMM_SERVICE # Still uses the injected global comm_service
    if TOOL_COMM_SERVICE is None:
        logger.error("Tool set_device_attribute: communication service not initialized.")
        return "Error: Device control service is not ready (tool not initialized)."

    # Delegate to the core logic function
    return await _set_device_attribute_core_logic(
        comm_service=TOOL_COMM_SERVICE,
        user_description=user_description,
        attribute=attribute,
        value=value,
        guessed_friendly_name=guessed_friendly_name
    )


async def set_device_attribute_directly( # This is what DefaultOfflineCommandProcessor calls
    comm_service: CommunicationServiceBase, # Takes comm_service directly
    device_friendly_name: str,
    attribute_name: str,
    value_str: str
) -> str:
    """
    Directly sets a device attribute, assuming device_friendly_name is already resolved.
    """
    # Delegate to the core logic function
    # Here, device_friendly_name IS the resolved name, so pass it as guessed_friendly_name
    # to prioritize it in the core logic's device finding step.
    # user_description can be the same as device_friendly_name or a placeholder.
    return await _set_device_attribute_core_logic(
        comm_service=comm_service,
        user_description=device_friendly_name, # Or a generic placeholder
        attribute=attribute_name,
        value=value_str,
        guessed_friendly_name=device_friendly_name # This ensures it's used directly
    )