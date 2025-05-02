# --- START OF FILE mqtt_client.py ---

import asyncio
import logging
import paho.mqtt.client as paho
import time
import json
import random
from typing import List, Dict, Any, Optional # Добавили типы
from settings import settings

logger = logging.getLogger(__name__)

# --- Структура для хранения информации об устройстве ---
DeviceInfo = Dict[str, Any] # Типаж для информации об устройстве

class MQTTClient:
    """
    Manages connection, publishing, and retrieving device capabilities from MQTT.
    """
    def __init__(self):
        self._client = None
        self._loop = None
        self._is_connected = False
        client_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        self._client_id = f"{settings.mqtt_broker.client_id_prefix}{client_suffix}"
        # Store device information {friendly_name: DeviceInfo}
        self._devices: Dict[str, DeviceInfo] = {}
        self._devices_topic = f"{settings.mqtt_broker.default_topic_base}/bridge/devices"
        self._device_list_ready = asyncio.Event()

    # --- _on_connect, _on_disconnect, _on_publish остаются как были ---
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"MQTT Client ({self._client_id}) connected successfully.")
            self._is_connected = True
            logger.info(f"MQTT Client subscribing to: {self._devices_topic}")
            client.subscribe(self._devices_topic)
        else:
            logger.error(f"MQTT Client ({self._client_id}) connection failed with code: {rc}.")
            self._is_connected = False
            self._device_list_ready.clear()

    def _on_disconnect(self, client, userdata, rc, properties=None):
        self._is_connected = False
        self._devices = {}
        self._device_list_ready.clear()
        logger.warning(f"MQTT Client ({self._client_id}) disconnected with result code {rc}.")

    def _on_publish(self, client, userdata, mid):
        logger.debug(f"MQTT message published successfully (MID: {mid})")


    # --- Обновленный _on_message для парсинга exposes ---
    def _on_message(self, client, userdata, msg):
        logger.debug(f"MQTT received message on topic '{msg.topic}'")
        if msg.topic == self._devices_topic:
            try:
                devices_payload = json.loads(msg.payload.decode())
                if isinstance(devices_payload, list):
                    new_devices = {}
                    for device_data in devices_payload:
                        if isinstance(device_data, dict) and device_data.get("friendly_name"):
                            friendly_name = device_data["friendly_name"]
                            # Сохраняем основную информацию и exposes
                            device_info: DeviceInfo = {
                                "friendly_name": friendly_name,
                                "model": device_data.get("model_id", "unknown"),
                                "type": device_data.get("type", "unknown"),
                                # Парсим и сохраняем exposes, если есть
                                "exposes": self._parse_exposes(device_data.get("definition", {}).get("exposes", [])),
                                # Можно добавить 'supported_features' для быстрого доступа, если нужно
                            }
                            new_devices[friendly_name] = device_info

                    logger.info(f"MQTT received and processed device list update. Found {len(new_devices)} devices with parsed exposes.")
                    self._devices = new_devices
                    if not self._device_list_ready.is_set() and new_devices: # Сигналим только если список не пуст
                        self._device_list_ready.set()
                else:
                    logger.warning(f"Received unexpected payload format on devices topic '{msg.topic}'. Expected a list.")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from devices topic '{msg.topic}'.")
            except Exception as e:
                logger.error(f"Error processing devices message: {e}", exc_info=True)
        else:
            logger.debug(f"MQTT received message on unhandled topic '{msg.topic}'.")

    # --- Новый метод для рекурсивного парсинга exposes ---
    def _parse_exposes(self, exposes_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Рекурсивно парсит список exposes из Zigbee2MQTT в словарь {feature_name: feature_details}.
        Обрабатывает вложенные структуры (например, для составных устройств).
        """
        parsed = {}
        if not isinstance(exposes_list, list):
            return {}

        for item in exposes_list:
            if not isinstance(item, dict): continue
            item_type = item.get("type")
            feature_name = item.get("property") # Часто имя свойства здесь
            endpoint = item.get("endpoint") # Иногда имя в endpoint

            # Имя может быть в 'property' или 'feature.name'
            if not feature_name and "features" in item: # Обработка вложенных features
                 nested_parsed = self._parse_exposes(item.get("features", []))
                 # Добавляем endpoint к вложенным именам для уникальности, если есть
                 if endpoint:
                      for name, details in nested_parsed.items():
                          parsed[f"{endpoint}_{name}"] = details
                 else:
                      parsed.update(nested_parsed)

            elif feature_name:
                 # Пропускаем системные 'linkquality'
                 if feature_name == 'linkquality': continue

                 feature_details = {
                     "type": item_type,
                     "name": feature_name, # Оригинальное имя свойства
                     "access": item.get("access", 7), # 7 = publish+set+get (битовая маска)
                     "description": item.get("description"),
                     "endpoint": endpoint
                 }
                 # Добавляем специфичные для типа поля
                 if item_type == "numeric":
                     feature_details["value_min"] = item.get("value_min")
                     feature_details["value_max"] = item.get("value_max")
                     feature_details["unit"] = item.get("unit")
                 elif item_type == "binary":
                     feature_details["value_on"] = item.get("value_on", "ON")
                     feature_details["value_off"] = item.get("value_off", "OFF")
                     feature_details["value_toggle"] = item.get("value_toggle", "TOGGLE")
                 elif item_type == "enum":
                     feature_details["values"] = item.get("values")
                 elif item_type == "light": # Тип light часто содержит вложенные фичи
                      nested_parsed = self._parse_exposes(item.get("features", []))
                      parsed.update(nested_parsed) # Добавляем вложенные (brightness, color_temp и т.д.)
                      # Также можем добавить сам 'state' для света
                      if 'state' not in parsed: # Добавляем state, если его еще нет
                          parsed['state'] = {
                              'type': 'binary', 'name': 'state', 'access': 7,
                              'value_on': 'ON', 'value_off': 'OFF', 'value_toggle': 'TOGGLE'
                          }

                 # Добавляем основную фичу в словарь
                 # Используем endpoint для уникальности, если он есть и имя не уникально
                 final_name = f"{endpoint}_{feature_name}" if endpoint and feature_name in parsed else feature_name
                 parsed[final_name] = feature_details

        return parsed


    # --- connect, _wait_for_connection, disconnect остаются как были ---
    async def connect(self):
        # ... (код без изменений) ...
        if self._is_connected: return
        self._loop = asyncio.get_running_loop()
        self._client = paho.Client(client_id=self._client_id, protocol=paho.MQTTv5)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_publish = self._on_publish
        if settings.mqtt_broker.username and settings.mqtt_broker.password:
            self._client.username_pw_set(settings.mqtt_broker.username, settings.mqtt_broker.password)
        elif settings.mqtt_broker.username:
             self._client.username_pw_set(settings.mqtt_broker.username)
        logger.info(f"MQTT Client ({self._client_id}) connecting...")
        try:
            self._client.loop_start()
            await self._loop.run_in_executor(None, self._client.connect, settings.mqtt_broker.host, settings.mqtt_broker.port, 60)
            connect_timeout = 10.0
            try: await asyncio.wait_for(self._wait_for_connection(), timeout=connect_timeout)
            except asyncio.TimeoutError:
                logger.error(f"MQTT Client ({self._client_id}) connection timed out.")
                await self.disconnect(); return
            if self._is_connected:
                 device_list_timeout = 20.0
                 logger.info(f"MQTT Client connected. Waiting up to {device_list_timeout}s for device list...")
                 try: await asyncio.wait_for(self._device_list_ready.wait(), timeout=device_list_timeout)
                 except asyncio.TimeoutError: logger.warning(f"MQTT timed out waiting for device list.")
        except Exception as e: logger.error(f"MQTT connection error: {e}", exc_info=True); await self.disconnect()

    async def _wait_for_connection(self):
        while not self._is_connected: await asyncio.sleep(0.1)

    async def disconnect(self):
        # ... (код без изменений, но с unsubcsribe) ...
        if self._client:
             logger.info(f"MQTT Client ({self._client_id}) disconnecting...")
             if self._is_connected:
                  try: self._client.unsubscribe(self._devices_topic)
                  except Exception as e: logger.warning(f"Error unsubscribing: {e}")
             self._client.loop_stop()
             try: self._client.disconnect()
             except Exception as e: logger.warning(f"Error during MQTT disconnect: {e}")
             self._is_connected = False; self._devices = {}; self._device_list_ready.clear()
             logger.info(f"MQTT Client ({self._client_id}) disconnected.")
        self._client = None; self._loop = None


    # --- _publish_async, schedule_publish остаются как были ---
    async def _publish_async(self, topic: str, payload: str, qos: int = 1, retain: bool = False):
        # ... (код без изменений) ...
        if not self._client or not self._is_connected: return False
        try:
            logger.info(f"MQTT Publishing to topic '{topic}': {payload}")
            msg_info = self._client.publish(topic, payload, qos=qos, retain=retain)
            return msg_info.rc == paho.MQTT_ERR_SUCCESS
        except Exception as e: logger.error(f"MQTT Error during publish: {e}", exc_info=True); return False

    def schedule_publish(self, topic: str, payload: str, qos: int = 1, retain: bool = False) -> bool:
        # ... (код без изменений) ...
        if not self._loop or not self._loop.is_running() or not self._is_connected: return False
        asyncio.run_coroutine_threadsafe(self._publish_async(topic, payload, qos, retain), self._loop)
        logger.debug(f"MQTT publish for topic '{topic}' scheduled.")
        return True

    # --- Обновленный get_device_friendly_names и новый get_device_capabilities ---
    def get_device_friendly_names(self) -> List[str]:
        """Возвращает список имен известных устройств."""
        # Небольшое изменение: возвращаем пустой список, если _devices пуст
        return list(self._devices.keys()) if self._is_connected else []

    def get_device_capabilities(self, friendly_name: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Возвращает словарь с возможностями ('exposes') для указанного устройства
        или None, если устройство не найдено или клиент не подключен.
        """
        if not self._is_connected:
            logger.warning(f"Cannot get capabilities for '{friendly_name}': MQTT client not connected.")
            return None
        device_info = self._devices.get(friendly_name)
        if not device_info:
            logger.warning(f"Device '{friendly_name}' not found in known devices.")
            return None
        return device_info.get("exposes")


# --- Глобальный экземпляр и функции startup/shutdown остаются без изменений ---
mqtt_client = MQTTClient()
async def startup_mqtt_client(): await mqtt_client.connect()
async def shutdown_mqtt_client(): await mqtt_client.disconnect()

# --- END OF FILE mqtt_client.py ---