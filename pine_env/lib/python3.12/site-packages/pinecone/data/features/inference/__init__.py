import warnings

from pinecone.inference import *

warnings.warn(
    "The module at `pinecone.data.features.inference` has moved to `pinecone.inference`. "
    "Please update your imports. "
    "This warning will become an error in a future version of the Pinecone Python SDK.",
    DeprecationWarning,
)
