from .pinecone_plugin import PineconePlugin
from typing import Type, TypeVar, Generic, Optional
from dataclasses import dataclass

T = TypeVar('T')
C = TypeVar('C')

@dataclass
class PluginMetadata(Generic[C, T]):
    target_object: str
    """
    What SDK object are you trying to extend with a plugin? For initial use cases, this
    will always be 'Pinecone', which refers to the Pinecone class in the python sdk.
    """

    namespace: str
    """
    This is the attribute of the target object that will hold a reference to the 
    instantiated plugin implementation.

    For example, you could have this plugin implementation:

    ```
    # pinecone_plugin/my_plugin/__init__.py
    class MyPlugin(PineconePlugin):
        def __init__(self, client_builder, config):
            pass

        def feature(self):
            return "feature-response"
    
    __installables__ = [PluginMetadata('Pinecone', 'foo', MyPlugin)]
    ```

    After installation in the python sdk, this plugin would be accessible like so:

    ```
    from pinecone import Pinecone

    pc = Pinecone(api_key='key')
    pc.foo.feature()  # returns 'feature-response'
    ```
    """

    implementation_class: Type[PineconePlugin[T, C]]
    """
    This should be a class extending the PineconePlugin abstract base class. This class should expose
    various feature methods with UX in mind, since it's the surface end-users will
    interact with. Often it will be a wrapper around a generated OpenAPI client object.
    """