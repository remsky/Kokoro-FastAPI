import warnings

from pinecone.db_control.models import *

warnings.warn(
    "The module at `pinecone.models` has moved to `pinecone.db_control.models`. "
    "This warning will become an error in a future version of the Pinecone Python SDK.",
    DeprecationWarning,
)
