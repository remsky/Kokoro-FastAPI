from typing import NamedTuple, Optional, List


class Pagination(NamedTuple):
    next: str


class ListResponse(NamedTuple):
    namespace: str
    vectors: List
    pagination: Optional[Pagination]
