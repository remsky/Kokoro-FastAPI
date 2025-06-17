import warnings

from pinecone.db_control import *

warnings.warn(
    "The module at `pinecone.control` has moved to `pinecone.db_control`. "
    "This warning will become an error in a future version of the Pinecone Python SDK.",
    DeprecationWarning,
)
