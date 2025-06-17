from dataclasses import dataclass, field
from typing import Optional, Dict, Union

from ..enums import PodIndexEnvironment, PodType


@dataclass(frozen=True)
class PodSpec:
    """
    PodSpec represents the configuration used to deploy a pod-based index.

    To learn more about the options for each configuration, please see [Understanding Indexes](https://docs.pinecone.io/docs/indexes)
    """

    environment: str
    """
    The environment where the pod index will be deployed. Example: 'us-east1-gcp'
    """

    replicas: Optional[int] = None
    """
    The number of replicas to deploy for the pod index. Default: 1
    """

    shards: Optional[int] = None
    """
    The number of shards to use. Shards are used to expand the amount of vectors you can store beyond the capacity of a single pod. Default: 1
    """

    pods: Optional[int] = None
    """
    Number of pods to deploy. Default: 1
    """

    pod_type: Optional[str] = "p1.x1"
    """
    This value combines pod type and pod size into a single string. This configuration is your main lever for vertical scaling.
    """

    metadata_config: Optional[Dict] = field(default_factory=dict)
    """
    If you are storing a lot of metadata, you can use this configuration to limit the fields which are indexed for search.

    This configuration should be a dictionary with the key 'indexed' and the value as a list of fields to index.

    Example:
    ```
    {'indexed': ['field1', 'field2']}
    ```
    """

    source_collection: Optional[str] = None
    """
    The name of the collection to use as the source for the pod index. This configuration is only used when creating a pod index from an existing collection.
    """

    def __init__(
        self,
        environment: Union[PodIndexEnvironment, str],
        pod_type: Union[PodType, str] = "p1.x1",
        replicas: Optional[int] = None,
        shards: Optional[int] = None,
        pods: Optional[int] = None,
        metadata_config: Optional[Dict] = None,
        source_collection: Optional[str] = None,
    ):
        object.__setattr__(
            self,
            "environment",
            environment.value if isinstance(environment, PodIndexEnvironment) else str(environment),
        )
        object.__setattr__(
            self, "pod_type", pod_type.value if isinstance(pod_type, PodType) else str(pod_type)
        )
        object.__setattr__(self, "replicas", replicas)
        object.__setattr__(self, "shards", shards)
        object.__setattr__(self, "pods", pods)
        object.__setattr__(
            self, "metadata_config", metadata_config if metadata_config is not None else {}
        )
        object.__setattr__(self, "source_collection", source_collection)

    def asdict(self) -> Dict:
        """
        Returns the PodSpec as a dictionary.
        """
        return {"pod": self.__dict__}
