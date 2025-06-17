from typing import TypedDict, Optional, List


class SearchQueryVectorTypedDict(TypedDict):
    """
    SearchQueryVector represents the vector values used to query.
    """

    values: Optional[List[float]]
    """
    The vector data included in the search request.
    Optional.
    """

    sparse_values: Optional[List[float]]
    """
    The sparse embedding values to search with.
    Optional.
    """

    sparse_indices: Optional[List[int]]
    """
    The sparse embedding indices to search with.
    Optional.
    """
