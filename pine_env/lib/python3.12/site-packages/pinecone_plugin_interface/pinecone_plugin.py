from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

C = TypeVar("C") # Config type
T = TypeVar("T") # openapi client instance

class PineconePlugin(ABC, Generic[C, T]):
    """
    The optional openapi_client parameter passed if the PluginMetadata passed
    to the plugin constructor has an openapi_client_class attribute.
    """
    @abstractmethod
    def __init__(self, config: C, openapi_client: Optional[T] = None):
        pass
