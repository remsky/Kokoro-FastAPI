from dataclasses import dataclass


@dataclass(frozen=True)
class ByocSpec:
    """
    ByocSpec represents the configuration used to deploy a BYOC (Bring Your Own Cloud) index.

    To learn more about the options for each configuration, please see [Understanding Indexes](https://docs.pinecone.io/docs/indexes)
    """

    environment: str
