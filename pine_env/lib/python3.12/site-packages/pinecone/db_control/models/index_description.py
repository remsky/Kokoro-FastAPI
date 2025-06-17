from typing import NamedTuple, Dict, Optional, Literal


class PodSpecDefinition(NamedTuple):
    replicas: int
    shards: int
    pods: int
    pod_type: str
    environment: str
    metadata_config: Optional[Dict]


class ServerlessSpecDefinition(NamedTuple):
    cloud: str
    region: str


PodKey = Literal["pod"]
PodSpec = Dict[PodKey, PodSpecDefinition]

ServerlessKey = Literal["serverless"]
ServerlessSpec = Dict[ServerlessKey, ServerlessSpecDefinition]
