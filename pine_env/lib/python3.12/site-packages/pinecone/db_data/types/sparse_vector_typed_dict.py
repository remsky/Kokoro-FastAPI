from typing import TypedDict, List


class SparseVectorTypedDict(TypedDict):
    indices: List[int]
    values: List[float]
