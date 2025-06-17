from .vector_metadata_dict import VectorMetadataTypedDict
from typing import Tuple, List

VectorTuple = Tuple[str, List[float]]
VectorTupleWithMetadata = Tuple[str, List[float], VectorMetadataTypedDict]
