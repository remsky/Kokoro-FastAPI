from typing import TypedDict, Optional, Union, Dict, Any
from .search_query_vector_typed_dict import SearchQueryVectorTypedDict


class SearchQueryTypedDict(TypedDict):
    """
    SearchQuery represents the query when searching within a specific namespace.
    """

    inputs: Dict[str, Any]
    """
    The input data to search with.
    Required.
    """

    top_k: int
    """
    The number of results to return with each search.
    Required.
    """

    filter: Optional[Dict[str, Any]]
    """
    The filter to apply to the search.
    Optional.
    """

    vector: Optional[Union[SearchQueryVectorTypedDict]]
    """
    The vector values to search with. If provided, it overwrites the inputs.
    """

    id: Optional[str]
    """
    The unique ID of the vector to be used as a query vector.
    """
