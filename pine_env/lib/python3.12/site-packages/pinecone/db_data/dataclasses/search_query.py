from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from .search_query_vector import SearchQueryVector
from ..types.search_query_vector_typed_dict import SearchQueryVectorTypedDict


@dataclass
class SearchQuery:
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

    filter: Optional[Dict[str, Any]] = None
    """
    The filter to apply to the search.
    Optional.
    """

    vector: Optional[Union[SearchQueryVectorTypedDict, SearchQueryVector]] = None
    """
    The vector values to search with. If provided, it overwrites the inputs.
    """

    id: Optional[str] = None
    """
    The unique ID of the vector to be used as a query vector.
    """

    def __post_init__(self):
        """
        Converts `vector` to a `SearchQueryVectorTypedDict` instance if an enum is provided.
        """
        if isinstance(self.vector, SearchQueryVector):
            self.vector = self.vector.as_dict()

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns the SearchQuery as a dictionary.
        """
        d = {
            "inputs": self.inputs,
            "top_k": self.top_k,
            "filter": self.filter,
            "vector": self.vector,
            "id": self.id,
        }
        return {k: v for k, v in d.items() if v is not None}
