import warnings

warnings.warn(
    "The module at `pinecone.data` has moved to `pinecone.db_data`. "
    "Please update your imports. "
    "This warning will become an error in a future version of the Pinecone Python SDK.",
    DeprecationWarning,
)
