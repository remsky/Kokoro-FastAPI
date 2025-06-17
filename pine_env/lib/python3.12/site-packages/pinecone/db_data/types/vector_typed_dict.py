from .sparse_vector_typed_dict import SparseVectorTypedDict
from typing import TypedDict, List


class VectorTypedDict(TypedDict, total=False):
    values: List[float]
    metadata: dict
    sparse_values: SparseVectorTypedDict
    id: str
