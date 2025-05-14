import asyncio
import json
import logging
import random
from typing import Dict, Any, List, Optional

import paho.mqtt.client as paho_mqtt # Use explicit import alias
from paho.mqtt.enums import CallbackAPIVersion

from .base import CommunicationServiceBase
from settings import MqttBrokerSettings # Specific settings

logger = logging.getLogger(__name__)

# Type alias from your mqtt_client.py
DeviceInfo = Dict[str, Any]

class MQTTService(CommunicationServiceBase):
    def __init__(self, config: Dict[str, Any]): # config is settings.mqtt_broker
        self.settings: MqttBrokerSettings = config
        self._client: Optional[paho_mqtt.Client] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None # Will be set in startup
        self._is_connected_event = asyncio.Event() # Renamed for clarity

        client_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        self._client_id = f"{self.settings.client_id_prefix}{client_suffix}"
        
        # Instance variables for device information
        self._devices: Dict[str, DeviceInfo] = {} # friendly_name -> DeviceInfo
        self._devices_topic = f"{self.settings.default_topic_base}/bridge/devices"
        self._device_list_ready_event = asyncio.Event()

        # If you need to map friendly names to IEEE addresses (common for Z2M commands)
        self._friendly_name_to_ieee_addr: Dict[str, str] = {}


    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"MQTT Service ({self._client_id}): Connected to broker successfully.")
            self._is_connected_event.set()
            # Subscribe to device topics for discovery
            client.subscribe(self._devices_topic)
            logger.info(f"MQTT Service ({self._client_id}): Subscribed to {self._devices_topic}")
        else:
            logger.error(f"MQTT Service ({self._client_id}): Failed to connect to broker, return code {rc}")
            self._is_connected_event.clear()
            self._device_list_ready_event.clear() # Also clear this

    def _on_disconnect(self, client, userdata, rc, properties=None):
        logger.warning(f"MQTT Service ({self._client_id}): Disconnected from broker with result code {rc}.")
        self._is_connected_event.clear()
        self._devices.clear()
        self._friendly_name_to_ieee_addr.clear()
        self._device_list_ready_event.clear()

    def _on_publish(self, client, userdata, mid, properties=None, reason_code=None): # Added reason_code for V2
        # V2 callback might have slightly different args, check Paho docs if issues.
        # This is mostly for debug.
        logger.debug(f"MQTT Service ({self._client_id}): Message published (MID: {mid})")


    def _on_message(self, client, userdata, msg: paho_mqtt.MQTTMessage):
        # Use call_soon_threadsafe to delegate processing to the asyncio loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._process_message_sync, msg.topic, msg.payload)
        else:
            logger.warning("MQTTService: Loop not available in _on_message, cannot process message.")

    def _process_message_sync(self, topic: str, payload_bytes: bytes):
        # This method is called by call_soon_threadsafe, so it runs in the asyncio event loop.
        # We can make it async internally to use await for locks if necessary,
        # but for simple dict updates, direct modification is often fine if _on_message
        # callbacks from Paho are serialized (which they usually are per client).
        # For safety with async modification:
        async def process_async_wrapper():
            payload_str = payload_bytes.decode('utf-8')
            logger.debug(f"MQTT Service ({self._client_id}): Received on '{topic}': {payload_str[:100]}...")
            
            if topic == self._devices_topic:
                try:
                    devices_payload = json.loads(payload_str)
                    if isinstance(devices_payload, list):
                        new_devices: Dict[str, DeviceInfo] = {}
                        new_friendly_name_to_ieee: Dict[str, str] = {}

                        for device_data in devices_payload:
                            if isinstance(device_data, dict) and device_data.get("friendly_name"):
                                friendly_name = device_data["friendly_name"]
                                ieee_address = device_data.get("ieee_address")

                                device_info: DeviceInfo = {
                                    "friendly_name": friendly_name,
                                    "ieee_address": ieee_address, # Store for command construction
                                    "model": device_data.get("model_id", device_data.get("model", "unknown")),
                                    "type": device_data.get("type", "unknown"),
                                    "supported": device_data.get("supported", False),
                                    "exposes": self._parse_exposes(device_data.get("definition", {}).get("exposes", [])),
                                    # Retain the full definition if needed for other purposes
                                    "definition": device_data.get("definition"),
                                    # Retain the original full data as well
                                    "_raw_data": device_data
                                }
                                new_devices[friendly_name] = device_info
                                if ieee_address: # Check if ieee_address exists
                                    new_friendly_name_to_ieee[friendly_name] = ieee_address
                        
                        # Atomically update the device stores
                        self._devices = new_devices
                        self._friendly_name_to_ieee_addr = new_friendly_name_to_ieee

                        logger.info(f"MQTT Service ({self._client_id}): Processed device list. {len(self._devices)} devices.")
                        if not self._device_list_ready_event.is_set() and self._devices:
                            self._device_list_ready_event.set()
                    else:
                        logger.warning(f"MQTT Service ({self._client_id}): Unexpected payload on '{topic}'. Expected list.")
                except json.JSONDecodeError:
                    logger.error(f"MQTT Service ({self._client_id}): JSON decode error on '{topic}'.")
                except Exception as e:
                    logger.error(f"MQTT Service ({self._client_id}): Error processing message from '{topic}': {e}", exc_info=True)
            else:
                logger.debug(f"MQTT Service ({self._client_id}): Message on unhandled topic '{topic}'.")

        if self._loop: # Ensure loop is available
            asyncio.create_task(process_async_wrapper())


    def _parse_exposes(self, exposes_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Recursively parses Zigbee2MQTT 'exposes' list into {feature_name: feature_details}.
        (Copied and adapted from your mqtt_client.py)
        """
        parsed = {}
        if not isinstance(exposes_list, list):
            return {}

        for item in exposes_list:
            if not isinstance(item, dict): continue
            
            item_type = item.get("type")
            # 'name' is sometimes used for high-level features, 'property' for specific ones
            # 'feature' itself can be a key for a list of sub-features
            feature_name = item.get("property") or item.get("name")
            endpoint = item.get("endpoint")

            if item_type == 'group' and "features" in item: # Handle groups by parsing their features
                nested_parsed = self._parse_exposes(item.get("features", []))
                # Prefix with group's endpoint if available and distinct
                if endpoint:
                    for name, details in nested_parsed.items():
                        parsed[f"{endpoint}_{name}"] = details # Or some other disambiguation
                else:
                    parsed.update(nested_parsed)
                continue # Skip further processing for this group item

            if not feature_name: # If no clear name/property, try to parse sub-features
                if "features" in item:
                    nested_parsed = self._parse_exposes(item.get("features", []))
                    if endpoint: # Prefix with endpoint if available
                        for name, details in nested_parsed.items():
                            parsed[f"{endpoint}_{name}"] = details
                    else:
                        parsed.update(nested_parsed)
                continue

            if feature_name == 'linkquality': continue # Skip system feature

            feature_details = {
                "type": item_type,
                "name": feature_name, # Original name
                "access": item.get("access", 7), 
                "description": item.get("description"),
                "endpoint": endpoint # Store endpoint for constructing command topics
            }

            if item_type == "numeric":
                feature_details.update({
                    "value_min": item.get("value_min"), "value_max": item.get("value_max"),
                    "value_step": item.get("value_step"), "unit": item.get("unit")
                })
            elif item_type == "binary":
                feature_details.update({
                    "value_on": item.get("value_on", "ON"), "value_off": item.get("value_off", "OFF"),
                    "value_toggle": item.get("value_toggle", "TOGGLE")
                })
            elif item_type == "enum":
                feature_details["values"] = item.get("values")
            elif item_type == "text":
                pass # No specific extra fields for text usually
            elif item_type == "light": # Light often has sub-features for brightness, color_temp, etc.
                if "features" in item:
                    nested_light_features = self._parse_exposes(item.get("features", []))
                    # Prefix light sub-features with the main light feature_name or endpoint for clarity
                    # e.g., "brightness" becomes "light_brightness" or if endpoint "L1_brightness"
                    prefix = f"{endpoint}_" if endpoint else f"{feature_name}_" # Choose a sensible prefix
                    
                    # Add the main light state itself if not already present through sub-features
                    # Typically, a light's primary control is its 'state'
                    # This might be implicitly handled if 'state' is a feature in item.get("features")
                    # For now, let's assume sub-features cover it or it's a top-level binary.
                    
                    for sub_name, sub_details in nested_light_features.items():
                        # Avoid double prefixing if sub_name already seems prefixed
                        final_sub_name = sub_name if sub_name.startswith(prefix.split('_')[0]) else f"{prefix}{sub_name}"
                        parsed[final_sub_name] = sub_details
                # Add a general state for the light itself if not covered by sub-features
                # This is a common pattern for Z2M lights.
                # The key "state" is often used.
                # If we add sub-features like "brightness", "color_temp", we also need a general "state" for on/off.
                # Check if a "state" feature was already parsed as a sub-feature.
                state_key_candidate = f"{endpoint}_state" if endpoint else "state"
                if not any(k.endswith("state") for k in parsed.keys()): # Simplified check
                     parsed[state_key_candidate] = {
                        'type': 'binary', 'name': 'state', 'access': 7, 'endpoint': endpoint,
                        'value_on': 'ON', 'value_off': 'OFF', 'value_toggle': 'TOGGLE',
                        'description': 'On/off state of the light'
                     }

            # Use endpoint to create a more unique key if necessary, esp. for multi-endpoint devices
            # However, feature_name from Z2M is usually unique enough for single device context.
            # If endpoint is present and helps disambiguate, use it.
            # Example: if two "switch" features exist on different endpoints.
            final_key_name = f"{endpoint}_{feature_name}" if endpoint else feature_name
            parsed[final_key_name] = feature_details
            
        return parsed

    async def startup(self):
        if self._is_connected_event.is_set():
            logger.info(f"MQTT Service ({self._client_id}): Already connected.")
            return

        self._loop = asyncio.get_running_loop() # Get loop here
        self._client = paho_mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id=self._client_id)
        # self._client.mqtt_version = paho_mqtt.MQTTv5 # Explicitly set if needed, Paho often negotiates
        
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_publish = self._on_publish # Added from your client

        if self.settings.username: # Handles username only or username+password
            self._client.username_pw_set(self.settings.username, self.settings.password)
        
        logger.info(f"MQTT Service ({self._client_id}): Attempting to connect to {self.settings.host}:{self.settings.port}")
        try:
            # Start Paho's network loop thread. This must be done before connect for some race conditions.
            self._client.loop_start()
            
            # Perform the blocking connect call in an executor thread
            await self._loop.run_in_executor(None, self._client.connect, self.settings.host, self.settings.port, 60)
            
            connect_timeout = 10.0 # Timeout for _is_connected_event
            try:
                await asyncio.wait_for(self._is_connected_event.wait(), timeout=connect_timeout)
                logger.info(f"MQTT Service ({self._client_id}): Connection established.")
                
                # Wait for the initial device list
                device_list_timeout = 20.0 
                logger.info(f"MQTT Service ({self._client_id}): Waiting up to {device_list_timeout}s for device list...")
                try:
                    await asyncio.wait_for(self._device_list_ready_event.wait(), timeout=device_list_timeout)
                    logger.info(f"MQTT Service ({self._client_id}): Device list received.")
                except asyncio.TimeoutError:
                    logger.warning(f"MQTT Service ({self._client_id}): Timed out waiting for initial device list. May operate with partial data.")
            except asyncio.TimeoutError:
                logger.error(f"MQTT Service ({self._client_id}): Timeout waiting for connection confirmation after connect() call.")
                await self.shutdown() # Attempt to clean up
                return # Exit if connection timed out
        except Exception as e:
            logger.error(f"MQTT Service ({self._client_id}): Connection failed: {e}", exc_info=True)
            await self.shutdown() # Attempt to clean up

    async def shutdown(self):
        if self._client:
            logger.info(f"MQTT Service ({self._client_id}): Disconnecting...")
            if self._is_connected_event.is_set(): # Check if connected before trying to unsubscribe
                try:
                    # Unsubscribe is blocking, run in executor if it causes issues
                    # For Paho, unsubscribe is usually quick.
                    self._client.unsubscribe(self._devices_topic)
                    logger.debug(f"MQTT Service ({self._client_id}): Unsubscribed from {self._devices_topic}")
                except Exception as e:
                    logger.warning(f"MQTT Service ({self._client_id}): Error unsubscribing: {e}")

            self._client.loop_stop() # Stop the network loop; True to wait for thread to finish
            try:
                # Disconnect is blocking
                await self._loop.run_in_executor(None, self._client.disconnect)
            except Exception as e:
                logger.warning(f"MQTT Service ({self._client_id}): Error during MQTT disconnect call: {e}")
            
            self._is_connected_event.clear()
            self._devices.clear()
            self._friendly_name_to_ieee_addr.clear()
            self._device_list_ready_event.clear()
            logger.info(f"MQTT Service ({self._client_id}): Disconnected.")
        self._client = None
        # self._loop = None # Don't nullify loop if other async tasks might use it

    async def publish(self, topic: str, payload: str, retain: bool = False, qos: int = 0):
        if not self._client or not self._is_connected_event.is_set():
            logger.error(f"MQTT Service ({self._client_id}): Not connected, cannot publish to '{topic}'.")
            return False # Indicate failure

        logger.debug(f"MQTT Service ({self._client_id}): Publishing to '{topic}': {payload}")
        try:
            # Paho's publish is non-blocking by default (returns MessageInfo)
            # To make it awaitable or ensure it's sent before proceeding (if critical),
            # one might use msg_info.wait_for_publish() in an executor.
            # For most cases, fire-and-forget is fine.
            msg_info = await self._loop.run_in_executor(
                None, 
                lambda: self._client.publish(topic, payload, qos, retain)
            )
            # msg_info = self._client.publish(topic, payload, qos, retain) # If executor not needed

            if msg_info.rc == paho_mqtt.MQTT_ERR_SUCCESS:
                # logger.debug(f"Publish to {topic} successful (MID: {msg_info.mid})")
                return True
            else:
                 logger.error(f"MQTT Service ({self._client_id}): Failed to publish to {topic}. RC: {msg_info.rc}")
                 return False
        except Exception as e:
            logger.error(f"MQTT Service ({self._client_id}): Error publishing to {topic}: {e}", exc_info=True)
            return False

    def get_device_friendly_names(self) -> List[str]:
        return list(self._devices.keys()) if self._is_connected_event.is_set() else []

    def get_device_capabilities(self, friendly_name: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Returns the parsed 'exposes' dictionary for the device.
        """
        if not self._is_connected_event.is_set():
            # logger.warning(f"Cannot get capabilities for '{friendly_name}': MQTT not connected.")
            return None
        device_info = self._devices.get(friendly_name)
        if not device_info:
            # logger.warning(f"Device '{friendly_name}' not found for capabilities.")
            return None
        return device_info.get("exposes") # This is the parsed dictionary

    def get_device_ieee_addr(self, friendly_name: str) -> Optional[str]:
        if not self._is_connected_event.is_set(): return None
        # Assumes _friendly_name_to_ieee_addr is populated in _on_message
        return self._friendly_name_to_ieee_addr.get(friendly_name)

    def get_full_device_info(self, friendly_name: str) -> Optional[DeviceInfo]: # DeviceInfo is Dict[str, Any]
        if not self._is_connected_event.is_set(): return None
        return self._devices.get(friendly_name) # _devices stores the comprehensive info

    @property
    def is_connected(self) -> bool:
        return self._is_connected_event.is_set()