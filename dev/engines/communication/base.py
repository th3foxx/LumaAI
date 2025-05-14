from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class CommunicationServiceBase(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    async def startup(self):
        """Initializes and connects the communication service."""
        pass

    @abstractmethod
    async def shutdown(self):
        """Disconnects and cleans up the communication service."""
        pass

    @abstractmethod
    async def publish(self, topic: str, payload: str, retain: bool = False, qos: int = 0) -> bool:
        """
        Publishes a message to a given topic.
        Returns True on successful queuing/dispatch, False otherwise.
        """
        pass

    @abstractmethod
    def get_device_friendly_names(self) -> List[str]:
        """Returns a list of known device friendly names."""
        pass

    @abstractmethod
    def get_device_capabilities(self, friendly_name: str) -> Optional[Dict[str, Any]]:
        """
        Returns a dictionary of capabilities (e.g., parsed "exposes" from Zigbee2MQTT)
        for the specified device. The structure of this dictionary depends on the
        underlying communication system and device type.
        """
        pass

    @abstractmethod
    def get_device_ieee_addr(self, friendly_name: str) -> Optional[str]:
        """
        Returns the IEEE address (or other unique hardware identifier) for the
        specified device, if applicable and available.
        """
        pass

    @abstractmethod
    def get_full_device_info(self, friendly_name: str) -> Optional[Dict[str, Any]]:
        """
        Returns all available information stored about a device, which might include
        raw data, model, type, capabilities, etc.
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the service is currently connected, False otherwise."""
        pass