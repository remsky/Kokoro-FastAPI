from enum import Enum


class Metric(Enum):
    """
    The metric specifies how Pinecone should calculate the distance between vectors when querying an index.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOTPRODUCT = "dotproduct"
