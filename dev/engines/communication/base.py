from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class CommunicationServiceBase(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        pass

    @abstractmethod
    async def startup(self):
        pass

    @abstractmethod
    async def shutdown(self):
        pass

    @abstractmethod
    async def publish(self, topic: str, payload: str, retain: bool = False, qos: int = 0):
        pass

    @abstractmethod
    def get_device_friendly_names(self) -> List[str]:
        pass

    @abstractmethod
    def get_device_capabilities(self, friendly_name: str) -> Optional[Dict[str, Any]]:
        pass

    # Add other necessary methods from mqtt_client