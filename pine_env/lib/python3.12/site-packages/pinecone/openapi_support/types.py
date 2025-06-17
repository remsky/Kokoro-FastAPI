from typing import TypedDict, Dict, Union


class PropertyValidationTypedDict(TypedDict, total=False):
    max_length: int
    min_length: int
    max_items: int
    min_items: int
    exclusive_maximum: Union[int, float]
    inclusive_maximum: Union[int, float]
    exclusive_minimum: Union[int, float]
    inclusive_minimum: Union[int, float]
    regex: Dict[str, str]
    multiple_of: int
