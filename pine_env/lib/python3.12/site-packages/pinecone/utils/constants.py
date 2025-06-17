import os

from .version import __version__

MAX_MSG_SIZE = 128 * 1024 * 1024

MAX_ID_LENGTH = int(os.getenv("PINECONE_MAX_ID_LENGTH", default="64"))

REQUEST_ID: str = "request_id"

CLIENT_VERSION = __version__
CLIENT_ID = f"python-client-{CLIENT_VERSION}"

REQUIRED_VECTOR_FIELDS = {"id"}
OPTIONAL_VECTOR_FIELDS = {"values", "sparse_values", "metadata"}

SOURCE_TAG = "source_tag"
