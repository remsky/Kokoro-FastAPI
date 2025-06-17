from dataclasses import dataclass
from typing import Union
from enum import Enum

from ..enums import CloudProvider, AwsRegion, GcpRegion, AzureRegion


@dataclass(frozen=True)
class ServerlessSpec:
    cloud: str
    region: str

    def __init__(
        self,
        cloud: Union[CloudProvider, str],
        region: Union[AwsRegion, GcpRegion, AzureRegion, str],
    ):
        # Convert Enums to their string values if necessary
        object.__setattr__(self, "cloud", cloud.value if isinstance(cloud, Enum) else str(cloud))
        object.__setattr__(
            self, "region", region.value if isinstance(region, Enum) else str(region)
        )

    def asdict(self):
        return {"serverless": {"cloud": self.cloud, "region": self.region}}
