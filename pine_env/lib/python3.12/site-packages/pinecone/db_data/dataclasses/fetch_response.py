from dataclasses import dataclass
from typing import Dict

from .vector import Vector


@dataclass
class FetchResponse:
    namespace: str
    vectors: Dict[str, Vector]
    usage: Dict[str, int]
