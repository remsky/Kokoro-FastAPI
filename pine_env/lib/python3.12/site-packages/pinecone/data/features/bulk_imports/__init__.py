import warnings

from pinecone.db_data.resources.asyncio.bulk_import_asyncio import *
from pinecone.db_data.resources.sync.bulk_import import *
from pinecone.db_data.resources.sync.bulk_import_request_factory import *


warnings.warn(
    "The module at `pinecone.data.features.bulk_import` has moved to `pinecone.db_data.features.bulk_import`. "
    "Please update your imports. "
    "This warning will become an error in a future version of the Pinecone Python SDK.",
    DeprecationWarning,
)
