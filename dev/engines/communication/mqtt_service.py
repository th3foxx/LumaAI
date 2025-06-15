# engines/communication/mqtt_service.py

import asyncio
import json
import logging
import random
from typing import Dict, Any, List, Optional, Callable, Coroutine

import paho.mqtt.client as paho_mqtt # Use explicit import alias
from paho.mqtt.enums import CallbackAPIVersion

from .base import CommunicationServiceBase
from settings import MqttBrokerSettings # Specific settings

logger = logging.getLogger(__name__)

# Type alias from your mqtt_client.py
DeviceInfo = Dict[str, Any]
# --- НОВОЕ: Тип для callback-функций подписки ---
SubscriptionCallback = Callable[[str, bytes, Dict], Coroutine[Any, Any, None]]


class MQTTService(CommunicationServiceBase):
    def __init__(self, config: MqttBrokerSettings): # --- ИЗМЕНЕНО: Явно указываем тип конфига ---
        self.settings: MqttBrokerSettings = config
        self._client: Optional[paho_mqtt.Client] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._is_connected_event = asyncio.Event()

        client_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        self._client_id = f"{self.settings.client_id_prefix}{client_suffix}"
        
        self._devices: Dict[str, DeviceInfo] = {}
        self._devices_topic = f"{self.settings.default_topic_base}/bridge/devices"
        self._device_list_ready_event = asyncio.Event()

        self._friendly_name_to_ieee_addr: Dict[str, str] = {}
        
        # --- НОВОЕ: Кэш состояний и механизм подписок ---
        self._device_states: Dict[str, Dict] = {} # Кэш для 'zigbee2mqtt/FRIENDLY_NAME'
        self._state_lock = asyncio.Lock()
        self._subscriptions: Dict[str, List[SubscriptionCallback]] = {} # topic -> list of async callbacks


    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"MQTT Service ({self._client_id}): Connected to broker successfully.")
            self._is_connected_event.set()
            # --- ИЗМЕНЕНО: Подписываемся на всё в z2m, чтобы ловить и состояния ---
            subscription_topic = f"{self.settings.default_topic_base}/#"
            client.subscribe(subscription_topic)
            logger.info(f"MQTT Service ({self._client_id}): Subscribed to {subscription_topic}")
        else:
            logger.error(f"MQTT Service ({self._client_id}): Failed to connect to broker, return code {rc}")
            self._is_connected_event.clear()
            self._device_list_ready_event.clear()

    def _on_disconnect(self, client, userdata, rc, properties=None):
        logger.warning(f"MQTT Service ({self._client_id}): Disconnected from broker with result code {rc}.")
        self._is_connected_event.clear()
        self._devices.clear()
        self._friendly_name_to_ieee_addr.clear()
        self._device_list_ready_event.clear()
        # --- НОВОЕ: Очищаем кэш состояний при дисконнекте ---
        self._device_states.clear()

    def _on_publish(self, client, userdata, mid, properties=None, reason_code=None):
        logger.debug(f"MQTT Service ({self._client_id}): Message published (MID: {mid})")


    def _on_message(self, client, userdata, msg: paho_mqtt.MQTTMessage):
        if self._loop:
            self._loop.call_soon_threadsafe(self._process_message_sync, msg.topic, msg.payload, getattr(msg, 'properties', None))
        else:
            logger.warning("MQTTService: Loop not available in _on_message, cannot process message.")

    def _process_message_sync(self, topic: str, payload_bytes: bytes, properties: Optional[Dict] = None):
        # --- ИЗМЕНЕНО: Передаем properties для кастомных подписчиков ---
        async def process_async_wrapper():
            # --- НОВОЕ: Логика вызова кастомных подписчиков ---
            # Проверяем, есть ли подписчики на этот топик (с поддержкой wildcard #)
            callbacks_to_run = self._subscriptions.get(topic, [])
            for sub_topic, sub_callbacks in self._subscriptions.items():
                if sub_topic.endswith('/#') and topic.startswith(sub_topic[:-1]):
                    callbacks_to_run.extend(sub_callbacks)
            
            if callbacks_to_run:
                for callback in callbacks_to_run:
                    try:
                        # Запускаем callback как корутину
                        await callback(topic, payload_bytes, properties or {})
                    except Exception as e:
                        logger.error(f"Error in custom MQTT subscription callback for topic '{topic}': {e}", exc_info=True)


            # --- ИЗМЕНЕНО: Основная логика теперь обрабатывает и состояния ---
            payload_str = payload_bytes.decode('utf-8')
            logger.debug(f"MQTT Service ({self._client_id}): Received on '{topic}': {payload_str[:100]}...")
            
            topic_parts = topic.split('/')
            
            # 1. Обработка списка всех устройств
            if topic == self._devices_topic:
                try:
                    # ... (ваш существующий код для self._devices_topic без изменений) ...
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
                                    "ieee_address": ieee_address,
                                    "model": device_data.get("model_id", device_data.get("model", "unknown")),
                                    "type": device_data.get("type", "unknown"),
                                    "supported": device_data.get("supported", False),
                                    "exposes": self._parse_exposes(device_data.get("definition", {}).get("exposes", [])),
                                    "definition": device_data.get("definition"),
                                    "_raw_data": device_data
                                }
                                new_devices[friendly_name] = device_info
                                if ieee_address:
                                    new_friendly_name_to_ieee[friendly_name] = ieee_address
                        
                        self._devices = new_devices
                        self._friendly_name_to_ieee_addr = new_friendly_name_to_ieee

                        logger.info(f"MQTT Service ({self._client_id}): Processed device list. {len(self._devices)} devices.")
                        if not self._device_list_ready_event.is_set() and self._devices:
                            self._device_list_ready_event.set()
                except Exception as e:
                    logger.error(f"MQTT Service ({self._client_id}): Error processing message from '{topic}': {e}", exc_info=True)

            # 2. --- НОВОЕ: Обработка топика состояния конкретного устройства ---
            elif len(topic_parts) == 2 and topic_parts[0] == self.settings.default_topic_base:
                device_name = topic_parts[1]
                if device_name in self._devices: # Убедимся, что это известное устройство
                    try:
                        state_payload = json.loads(payload_str)
                        async with self._state_lock:
                            if device_name not in self._device_states:
                                self._device_states[device_name] = {}
                            self._device_states[device_name].update(state_payload)
                        logger.debug(f"State cache updated for '{device_name}': {self._device_states[device_name]}")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from state topic '{topic}'")
                    except Exception as e:
                        logger.error(f"Error updating state cache for '{device_name}': {e}", exc_info=True)

        if self._loop:
            asyncio.create_task(process_async_wrapper())


    def _parse_exposes(self, exposes_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        # ... (ваш существующий код _parse_exposes без изменений) ...
        parsed = {}
        if not isinstance(exposes_list, list):
            return {}

        for item in exposes_list:
            if not isinstance(item, dict): continue
            
            item_type = item.get("type")
            feature_name = item.get("property") or item.get("name")
            endpoint = item.get("endpoint")

            if item_type == 'group' and "features" in item:
                nested_parsed = self._parse_exposes(item.get("features", []))
                if endpoint:
                    for name, details in nested_parsed.items():
                        parsed[f"{endpoint}_{name}"] = details
                else:
                    parsed.update(nested_parsed)
                continue

            if not feature_name:
                if "features" in item:
                    nested_parsed = self._parse_exposes(item.get("features", []))
                    if endpoint:
                        for name, details in nested_parsed.items():
                            parsed[f"{endpoint}_{name}"] = details
                    else:
                        parsed.update(nested_parsed)
                continue

            if feature_name == 'linkquality': continue

            feature_details = {
                "type": item_type,
                "name": feature_name,
                "access": item.get("access", 7), 
                "description": item.get("description"),
                "endpoint": endpoint
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
            elif item_type == "light":
                if "features" in item:
                    nested_light_features = self._parse_exposes(item.get("features", []))
                    prefix = f"{endpoint}_" if endpoint else f"{feature_name}_"
                    
                    for sub_name, sub_details in nested_light_features.items():
                        final_sub_name = sub_name if sub_name.startswith(prefix.split('_')[0]) else f"{prefix}{sub_name}"
                        parsed[final_sub_name] = sub_details
                state_key_candidate = f"{endpoint}_state" if endpoint else "state"
                if not any(k.endswith("state") for k in parsed.keys()):
                     parsed[state_key_candidate] = {
                        'type': 'binary', 'name': 'state', 'access': 7, 'endpoint': endpoint,
                        'value_on': 'ON', 'value_off': 'OFF', 'value_toggle': 'TOGGLE',
                        'description': 'On/off state of the light'
                     }

            final_key_name = f"{endpoint}_{feature_name}" if endpoint else feature_name
            parsed[final_key_name] = feature_details
            
        return parsed

    async def startup(self):
        if self._is_connected_event.is_set():
            logger.info(f"MQTT Service ({self._client_id}): Already connected.")
            return

        self._loop = asyncio.get_running_loop()
        self._client = paho_mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id=self._client_id)
        
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_publish = self._on_publish

        if self.settings.username:
            self._client.username_pw_set(self.settings.username, self.settings.password)
        
        logger.info(f"MQTT Service ({self._client_id}): Attempting to connect to {self.settings.host}:{self.settings.port}")
        try:
            self._client.loop_start()
            await self._loop.run_in_executor(None, self._client.connect, self.settings.host, self.settings.port, 60)
            
            connect_timeout = 10.0
            try:
                await asyncio.wait_for(self._is_connected_event.wait(), timeout=connect_timeout)
                logger.info(f"MQTT Service ({self._client_id}): Connection established.")
                
                device_list_timeout = 20.0 
                logger.info(f"MQTT Service ({self._client_id}): Waiting up to {device_list_timeout}s for device list...")
                try:
                    await asyncio.wait_for(self._device_list_ready_event.wait(), timeout=device_list_timeout)
                    logger.info(f"MQTT Service ({self._client_id}): Device list received.")
                except asyncio.TimeoutError:
                    logger.warning(f"MQTT Service ({self._client_id}): Timed out waiting for initial device list. May operate with partial data.")
            except asyncio.TimeoutError:
                logger.error(f"MQTT Service ({self._client_id}): Timeout waiting for connection confirmation after connect() call.")
                await self.shutdown()
                return
        except Exception as e:
            logger.error(f"MQTT Service ({self._client_id}): Connection failed: {e}", exc_info=True)
            await self.shutdown()

    async def shutdown(self):
        if self._client:
            logger.info(f"MQTT Service ({self._client_id}): Disconnecting...")
            if self._is_connected_event.is_set():
                try:
                    # --- ИЗМЕНЕНО: Отписываемся от общего топика ---
                    self._client.unsubscribe(f"{self.settings.default_topic_base}/#")
                    logger.debug(f"MQTT Service ({self._client_id}): Unsubscribed from {self.settings.default_topic_base}/#")
                except Exception as e:
                    logger.warning(f"MQTT Service ({self._client_id}): Error unsubscribing: {e}")

            self._client.loop_stop()
            try:
                await self._loop.run_in_executor(None, self._client.disconnect)
            except Exception as e:
                logger.warning(f"MQTT Service ({self._client_id}): Error during MQTT disconnect call: {e}")
            
            self._is_connected_event.clear()
            self._devices.clear()
            self._friendly_name_to_ieee_addr.clear()
            self._device_list_ready_event.clear()
            # --- НОВОЕ: Очищаем кэш и подписки ---
            self._device_states.clear()
            self._subscriptions.clear()
            logger.info(f"MQTT Service ({self._client_id}): Disconnected.")
        self._client = None

    async def publish(self, topic: str, payload: str, retain: bool = False, qos: int = 0) -> bool: # --- ИЗМЕНЕНО: Добавлен тип возврата ---
        if not self._client or not self._is_connected_event.is_set():
            logger.error(f"MQTT Service ({self._client_id}): Not connected, cannot publish to '{topic}'.")
            return False

        logger.debug(f"MQTT Service ({self._client_id}): Publishing to '{topic}': {payload}")
        try:
            msg_info = await self._loop.run_in_executor(
                None, 
                lambda: self._client.publish(topic, payload, qos, retain)
            )
            if msg_info.rc == paho_mqtt.MQTT_ERR_SUCCESS:
                return True
            else:
                 logger.error(f"MQTT Service ({self._client_id}): Failed to publish to {topic}. RC: {msg_info.rc}")
                 return False
        except Exception as e:
            logger.error(f"MQTT Service ({self._client_id}): Error publishing to {topic}: {e}", exc_info=True)
            return False

    # --- НОВЫЕ ПУБЛИЧНЫЕ МЕТОДЫ ---

    async def subscribe(self, topic: str, callback: SubscriptionCallback):
        """Allows other services to subscribe to MQTT topics."""
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
        self._subscriptions[topic].append(callback)
        logger.info(f"New subscription added for topic '{topic}' by {callback.__name__}")

    async def get_device_state(self, friendly_name: str) -> Optional[Dict]:
        """Public method to get the cached state of a device."""
        async with self._state_lock:
            return self._device_states.get(friendly_name)

    # --- Существующие публичные методы без изменений ---

    def get_device_friendly_names(self) -> List[str]:
        return list(self._devices.keys()) if self._is_connected_event.is_set() else []

    def get_device_capabilities(self, friendly_name: str) -> Optional[Dict[str, Dict[str, Any]]]:
        if not self._is_connected_event.is_set():
            return None
        device_info = self._devices.get(friendly_name)
        if not device_info:
            return None
        return device_info.get("exposes")

    def get_device_ieee_addr(self, friendly_name: str) -> Optional[str]:
        if not self._is_connected_event.is_set(): return None
        return self._friendly_name_to_ieee_addr.get(friendly_name)

    def get_full_device_info(self, friendly_name: str) -> Optional[DeviceInfo]:
        if not self._is_connected_event.is_set(): return None
        return self._devices.get(friendly_name)

    @property
    def is_connected(self) -> bool:
        return self._is_connected_event.is_set()