from enum import Enum


class VectorType(Enum):
    """
    VectorType is used to specifiy the type of vector you will store in the index.

    Dense vectors are used to store dense embeddings, which are vectors with non-zero values in most of the dimensions.

    Sparse vectors are used to store sparse embeddings, which allow vectors with zero values in most of the dimensions to be represented concisely.
    """

    DENSE = "dense"
    SPARSE = "sparse"
