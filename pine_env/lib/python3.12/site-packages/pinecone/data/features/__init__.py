import warnings

from .bulk_imports import *
from .inference import *


warnings.warn(
    "The module at `pinecone.data.features` has been removed. Code has been refactored and integrated into other parts of the client. "
    "This warning will become an error in a future version of the Pinecone Python SDK.",
    DeprecationWarning,
)
