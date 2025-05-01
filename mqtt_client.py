# --- START OF FILE mqtt_client.py ---

import asyncio
import logging
import paho.mqtt.client as paho
from typing import List
import json
import random
from settings import settings

logger = logging.getLogger(__name__)

class MQTTClient:
    """
    Manages the connection and publishing to the MQTT broker asynchronously.
    Provides methods to get device list and schedule publishing.
    """
    def __init__(self):
        self._client = None
        self._loop = None
        self._is_connected = False
        # Generate a unique client ID suffix for this instance
        client_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        self._client_id = f"{settings.mqtt_broker.client_id_prefix}{client_suffix}"
        # Store device information {friendly_name: device_info_dict}
        self._devices = {}
        self._devices_topic = f"{settings.mqtt_broker.default_topic_base}/bridge/devices"
        self._device_list_ready = asyncio.Event() # Event to signal when the list is first received

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback when connection to MQTT broker is established."""
        if rc == 0:
            logger.info(f"MQTT Client ({self._client_id}) connected successfully to {settings.mqtt_broker.host}:{settings.mqtt_broker.port}")
            self._is_connected = True
            # Subscribe to the devices topic upon connection
            logger.info(f"MQTT Client subscribing to: {self._devices_topic}")
            client.subscribe(self._devices_topic)
            # You could subscribe to other topics here if needed
        else:
            logger.error(f"MQTT Client ({self._client_id}) connection failed with code: {rc}. Check broker address, port, credentials.")
            self._is_connected = False
            self._device_list_ready.clear() # Reset event on disconnect

    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Callback when disconnected from MQTT broker."""
        self._is_connected = False
        self._devices = {} # Clear device list on disconnect
        self._device_list_ready.clear()
        logger.warning(f"MQTT Client ({self._client_id}) disconnected with result code {rc}. Will attempt to reconnect...")
        # paho-mqtt loop_start handles basic reconnection

    def _on_message(self, client, userdata, msg):
        """Callback for received messages."""
        logger.debug(f"MQTT received message on topic '{msg.topic}'")
        # Check if it's the devices list update
        if msg.topic == self._devices_topic:
            try:
                devices_payload = json.loads(msg.payload.decode())
                if isinstance(devices_payload, list):
                    new_devices = {}
                    for device_info in devices_payload:
                        # We need devices that can be controlled (have a friendly_name)
                        if isinstance(device_info, dict) and device_info.get("friendly_name"):
                            friendly_name = device_info["friendly_name"]
                            # Store relevant info, e.g., friendly_name and maybe supported features
                            new_devices[friendly_name] = {
                                "friendly_name": friendly_name,
                                # You can add more fields here if needed later
                                # "type": device_info.get("type"),
                                # "supported_features": device_info.get("definition", {}).get("exposes")
                            }
                    logger.info(f"MQTT received and processed device list update. Found {len(new_devices)} devices.")
                    self._devices = new_devices
                    if not self._device_list_ready.is_set():
                        self._device_list_ready.set() # Signal that the list is ready
                else:
                     logger.warning(f"Received unexpected payload format on devices topic '{msg.topic}'. Expected a list.")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON payload from devices topic '{msg.topic}'. Payload: {msg.payload[:100]}...")
            except Exception as e:
                 logger.error(f"Error processing devices message from topic '{msg.topic}': {e}", exc_info=True)
        else:
            # Handle other subscribed topics if any
            logger.debug(f"MQTT received message on unhandled topic '{msg.topic}': {msg.payload.decode()}")


    def _on_publish(self, client, userdata, mid):
        """Callback when a message is successfully published."""
        logger.debug(f"MQTT message published successfully (MID: {mid})")

    async def connect(self):
        """Establishes connection to the MQTT broker and waits for initial device list."""
        if self._is_connected:
            logger.warning(f"MQTT Client ({self._client_id}) already connected.")
            return

        self._loop = asyncio.get_running_loop()
        self._client = paho.Client(client_id=self._client_id, protocol=paho.MQTTv5)

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_publish = self._on_publish

        if settings.mqtt_broker.username and settings.mqtt_broker.password:
            self._client.username_pw_set(settings.mqtt_broker.username, settings.mqtt_broker.password)
            logger.info(f"MQTT Client ({self._client_id}) configured with username.")
        elif settings.mqtt_broker.username:
             logger.warning(f"MQTT Client ({self._client_id}) username provided but no password.")
             self._client.username_pw_set(settings.mqtt_broker.username)

        logger.info(f"MQTT Client ({self._client_id}) attempting to connect to {settings.mqtt_broker.host}:{settings.mqtt_broker.port}...")
        try:
            self._client.loop_start()
            await self._loop.run_in_executor(
                None,
                self._client.connect,
                settings.mqtt_broker.host,
                settings.mqtt_broker.port,
                60
            )
            # Wait for connection OR timeout
            connect_timeout = 10.0
            try:
                await asyncio.wait_for(self._wait_for_connection(), timeout=connect_timeout)
            except asyncio.TimeoutError:
                logger.error(f"MQTT Client ({self._client_id}) connection timed out after {connect_timeout}s.")
                await self.disconnect() # Clean up
                return # Stop further processing

            if self._is_connected:
                 # Now wait for the initial device list
                 device_list_timeout = 20.0
                 logger.info(f"MQTT Client ({self._client_id}) connected. Waiting up to {device_list_timeout}s for initial device list...")
                 try:
                     await asyncio.wait_for(self._device_list_ready.wait(), timeout=device_list_timeout)
                     logger.info(f"MQTT Client ({self._client_id}) initial device list received.")
                 except asyncio.TimeoutError:
                      logger.warning(f"MQTT Client ({self._client_id}) timed out waiting for initial device list. Device control might be unreliable until list is received.")
                      # Continue running, but the list might be empty initially.

        except ConnectionRefusedError:
             logger.error(f"MQTT Connection Refused: Check if the broker is running at {settings.mqtt_broker.host}:{settings.mqtt_broker.port} and accessible.")
             if self._client: self._client.loop_stop()
        except Exception as e:
            logger.error(f"MQTT Client ({self._client_id}) connection error: {e}", exc_info=True)
            if self._client: self._client.loop_stop()

    async def _wait_for_connection(self):
        """Helper coroutine to wait until _is_connected is True."""
        while not self._is_connected:
            await asyncio.sleep(0.1) # Check periodically

    async def disconnect(self):
        """Disconnects from the MQTT broker."""
        if self._client: # Check if client exists
             logger.info(f"MQTT Client ({self._client_id}) disconnecting...")
             # Unsubscribe cleanly if connected
             if self._is_connected:
                  try:
                      self._client.unsubscribe(self._devices_topic)
                  except Exception as e:
                      logger.warning(f"Error unsubscribing from {self._devices_topic}: {e}")
             self._client.loop_stop()
             # Disconnect can sometimes block, run in executor if issues arise
             try:
                  self._client.disconnect()
             except Exception as e:
                  logger.warning(f"Error during MQTT disconnect: {e}")

             self._is_connected = False
             self._devices = {}
             self._device_list_ready.clear()
             logger.info(f"MQTT Client ({self._client_id}) disconnected.")
        self._client = None
        self._loop = None

    async def _publish_async(self, topic: str, payload: str, qos: int = 1, retain: bool = False):
        # ... (keep this method as it was) ...
        if not self._client or not self._is_connected:
            logger.error(f"MQTT Client ({self._client_id}) cannot publish: Not connected.")
            return False

        try:
            logger.info(f"MQTT Publishing to topic '{topic}': {payload}")
            msg_info = self._client.publish(topic, payload, qos=qos, retain=retain)
            if msg_info.rc == paho.MQTT_ERR_SUCCESS:
                 logger.debug(f"MQTT Publish command successful (MID: {msg_info.mid})")
                 return True
            else:
                 logger.error(f"MQTT Publish command failed with error code: {msg_info.rc}")
                 return False
        except Exception as e:
            logger.error(f"MQTT Error during publish to '{topic}': {e}", exc_info=True)
            return False

    def schedule_publish(self, topic: str, payload: str, qos: int = 1, retain: bool = False) -> bool:
        # ... (keep this method as it was) ...
        if not self._loop or not self._loop.is_running():
            logger.error("MQTT Cannot schedule publish: Event loop not available.")
            return False
        if not self._is_connected:
            logger.error("MQTT Cannot schedule publish: Client not connected.")
            return False

        future = asyncio.run_coroutine_threadsafe(
            self._publish_async(topic, payload, qos, retain),
            self._loop
        )
        logger.debug(f"MQTT publish for topic '{topic}' scheduled.")
        return True

    # --- New method to get device names ---
    def get_device_friendly_names(self) -> List[str]:
        """
        Synchronously returns the list of known device friendly names.
        Returns an empty list if the client is not connected or the list hasn't been received yet.
        """
        if not self._is_connected or not self._devices:
            if not self._is_connected:
                 logger.warning("Attempted to get device names, but MQTT client is not connected.")
            elif not self._devices:
                 logger.warning("Attempted to get device names, but the list is empty (possibly not received yet).")
            return []
        return list(self._devices.keys())

# --- Global MQTT Client Instance ---
mqtt_client = MQTTClient()

# --- Functions for FastAPI startup/shutdown ---
async def startup_mqtt_client():
    """Connects the MQTT client during FastAPI startup."""
    await mqtt_client.connect()

async def shutdown_mqtt_client():
    """Disconnects the MQTT client during FastAPI shutdown."""
    await mqtt_client.disconnect()

# --- END OF FILE mqtt_client.py ---