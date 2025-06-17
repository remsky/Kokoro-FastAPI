from typing import TypedDict, Dict, Union
from pinecone.db_control.enums import Metric
from pinecone.inference import EmbedModel


class CreateIndexForModelEmbedTypedDict(TypedDict):
    model: Union[EmbedModel, str]
    field_map: Dict
    metric: Union[Metric, str]
    read_parameters: Dict
    write_parameters: Dict
