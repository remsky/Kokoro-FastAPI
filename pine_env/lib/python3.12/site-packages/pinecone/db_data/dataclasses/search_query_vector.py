from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SearchQueryVector:
    """
    SearchQueryVector represents the vector values used to query.
    """

    values: Optional[List[float]] = None
    """
    The vector data included in the search request.
    Optional.
    """

    sparse_values: Optional[List[float]] = None
    """
    The sparse embedding values to search with.
    Optional.
    """

    sparse_indices: Optional[List[int]] = None
    """
    The sparse embedding indices to search with.
    Optional.
    """

    def as_dict(self) -> dict:
        """
        Returns the SearchQueryVector as a dictionary.
        """
        d = {
            "values": self.values,
            "sparse_values": self.sparse_values,
            "sparse_indices": self.sparse_indices,
        }
        return {k: v for k, v in d.items() if v is not None}
