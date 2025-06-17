from packaging import version
from pinecone_plugin_interface import PluginMetadata
from .assistant import Assistant


try:
    import pinecone
except ImportError as e:
    raise ImportError(
        "This assistant plugin requires the Pinecone SDK to be installed. "
        "Please install the Pinecone SDK by running `pip install pinecone`"
    )

# TODO: We can remove this soon, 6.0.1 is already out. Just to be nice to the unfortunate user in the meantime.
if version.parse(pinecone.__version__) == version.parse('6.0.0'):
    raise ImportError(
        "This assistant plugin version is not compatible with Pinecone SDK version 6.0.0. "
        "Please upgrade the Pinecone SDK version by running `pip install --upgrade pinecone`"
    )

__installables__ = [
    PluginMetadata(
        target_object="Pinecone",
        namespace="assistant",
        implementation_class=Assistant
    ),
]
