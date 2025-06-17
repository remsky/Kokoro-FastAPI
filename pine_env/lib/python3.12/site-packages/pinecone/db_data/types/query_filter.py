from typing import Literal, Dict, List, Union

FieldValue = Union[str, int, float, bool]

ExactMatchFilter = Dict[str, FieldValue]

EqFilter = Dict[Literal["$eq"], FieldValue]
NeFilter = Dict[Literal["$ne"], FieldValue]

NumericFieldValue = Union[int, float]
GtFilter = Dict[Literal["$gt"], NumericFieldValue]
GteFilter = Dict[Literal["$gte"], NumericFieldValue]
LtFilter = Dict[Literal["$lt"], NumericFieldValue]
LteFilter = Dict[Literal["$lte"], NumericFieldValue]

InFilter = Dict[Literal["$in"], List[FieldValue]]
NinFilter = Dict[Literal["$nin"], List[FieldValue]]


SimpleFilter = Union[
    ExactMatchFilter,
    EqFilter,
    NeFilter,
    GtFilter,
    GteFilter,
    LtFilter,
    LteFilter,
    InFilter,
    NinFilter,
]
AndFilter = Dict[Literal["$and"], List[SimpleFilter]]

FilterTypedDict = Union[SimpleFilter, AndFilter]
