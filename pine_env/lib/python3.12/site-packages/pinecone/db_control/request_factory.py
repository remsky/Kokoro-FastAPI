import logging
from typing import Optional, Dict, Any, Union
from enum import Enum

from pinecone.utils import parse_non_empty_args, convert_enum_to_string

from pinecone.core.openapi.db_control.model.create_collection_request import CreateCollectionRequest
from pinecone.core.openapi.db_control.model.create_index_for_model_request import (
    CreateIndexForModelRequest,
)
from pinecone.core.openapi.db_control.model.create_index_for_model_request_embed import (
    CreateIndexForModelRequestEmbed,
)
from pinecone.core.openapi.db_control.model.create_index_request import CreateIndexRequest
from pinecone.core.openapi.db_control.model.configure_index_request import ConfigureIndexRequest
from pinecone.core.openapi.db_control.model.configure_index_request_spec import (
    ConfigureIndexRequestSpec,
)
from pinecone.core.openapi.db_control.model.configure_index_request_spec_pod import (
    ConfigureIndexRequestSpecPod,
)
from pinecone.core.openapi.db_control.model.deletion_protection import (
    DeletionProtection as DeletionProtectionModel,
)
from pinecone.core.openapi.db_control.model.index_spec import IndexSpec
from pinecone.core.openapi.db_control.model.index_tags import IndexTags
from pinecone.core.openapi.db_control.model.serverless_spec import (
    ServerlessSpec as ServerlessSpecModel,
)
from pinecone.core.openapi.db_control.model.byoc_spec import ByocSpec as ByocSpecModel
from pinecone.core.openapi.db_control.model.pod_spec import PodSpec as PodSpecModel
from pinecone.core.openapi.db_control.model.pod_spec_metadata_config import PodSpecMetadataConfig
from pinecone.core.openapi.db_control.model.create_index_from_backup_request import (
    CreateIndexFromBackupRequest,
)
from pinecone.db_control.models import ServerlessSpec, PodSpec, ByocSpec, IndexModel, IndexEmbed

from pinecone.db_control.enums import (
    Metric,
    VectorType,
    DeletionProtection,
    PodType,
    CloudProvider,
    AwsRegion,
    GcpRegion,
    AzureRegion,
)
from .types import CreateIndexForModelEmbedTypedDict


logger = logging.getLogger(__name__)
""" @private """


class PineconeDBControlRequestFactory:
    """
    @private

    This class facilitates translating user inputs into request objects.
    """

    @staticmethod
    def __parse_tags(tags: Optional[Dict[str, str]]) -> IndexTags:
        if tags is None:
            return IndexTags()
        else:
            return IndexTags(**tags)

    @staticmethod
    def __parse_deletion_protection(
        deletion_protection: Union[DeletionProtection, str],
    ) -> DeletionProtectionModel:
        deletion_protection = convert_enum_to_string(deletion_protection)
        if deletion_protection in ["enabled", "disabled"]:
            return DeletionProtectionModel(deletion_protection)
        else:
            raise ValueError("deletion_protection must be either 'enabled' or 'disabled'")

    @staticmethod
    def __parse_index_spec(spec: Union[Dict, ServerlessSpec, PodSpec, ByocSpec]) -> IndexSpec:
        if isinstance(spec, dict):
            if "serverless" in spec:
                spec["serverless"]["cloud"] = convert_enum_to_string(spec["serverless"]["cloud"])
                spec["serverless"]["region"] = convert_enum_to_string(spec["serverless"]["region"])

                index_spec = IndexSpec(serverless=ServerlessSpecModel(**spec["serverless"]))
            elif "pod" in spec:
                spec["pod"]["environment"] = convert_enum_to_string(spec["pod"]["environment"])
                args_dict = parse_non_empty_args(
                    [
                        ("environment", spec["pod"].get("environment")),
                        ("metadata_config", spec["pod"].get("metadata_config")),
                        ("replicas", spec["pod"].get("replicas")),
                        ("shards", spec["pod"].get("shards")),
                        ("pods", spec["pod"].get("pods")),
                        ("source_collection", spec["pod"].get("source_collection")),
                    ]
                )
                if args_dict.get("metadata_config"):
                    args_dict["metadata_config"] = PodSpecMetadataConfig(
                        indexed=args_dict["metadata_config"].get("indexed", None)
                    )
                index_spec = IndexSpec(pod=PodSpecModel(**args_dict))
            elif "byoc" in spec:
                index_spec = IndexSpec(byoc=ByocSpecModel(**spec["byoc"]))
            else:
                raise ValueError("spec must contain either 'serverless', 'pod', or 'byoc' key")
        elif isinstance(spec, ServerlessSpec):
            index_spec = IndexSpec(
                serverless=ServerlessSpecModel(cloud=spec.cloud, region=spec.region)
            )
        elif isinstance(spec, PodSpec):
            args_dict = parse_non_empty_args(
                [
                    ("replicas", spec.replicas),
                    ("shards", spec.shards),
                    ("pods", spec.pods),
                    ("source_collection", spec.source_collection),
                ]
            )
            if spec.metadata_config:
                args_dict["metadata_config"] = PodSpecMetadataConfig(
                    indexed=spec.metadata_config.get("indexed", None)
                )

            index_spec = IndexSpec(
                pod=PodSpecModel(environment=spec.environment, pod_type=spec.pod_type, **args_dict)
            )
        elif isinstance(spec, ByocSpec):
            args_dict = parse_non_empty_args([("environment", spec.environment)])
            index_spec = IndexSpec(byoc=ByocSpecModel(**args_dict))
        else:
            raise TypeError("spec must be of type dict, ServerlessSpec, PodSpec, or ByocSpec")

        return index_spec

    @staticmethod
    def create_index_request(
        name: str,
        spec: Union[Dict, ServerlessSpec, PodSpec, ByocSpec],
        dimension: Optional[int] = None,
        metric: Optional[Union[Metric, str]] = Metric.COSINE,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        vector_type: Optional[Union[VectorType, str]] = VectorType.DENSE,
        tags: Optional[Dict[str, str]] = None,
    ) -> CreateIndexRequest:
        if metric is not None:
            metric = convert_enum_to_string(metric)
        if vector_type is not None:
            vector_type = convert_enum_to_string(vector_type)
        if deletion_protection is not None:
            dp = PineconeDBControlRequestFactory.__parse_deletion_protection(deletion_protection)
        else:
            dp = None

        tags_obj = PineconeDBControlRequestFactory.__parse_tags(tags)
        index_spec = PineconeDBControlRequestFactory.__parse_index_spec(spec)

        if vector_type == VectorType.SPARSE.value and dimension is not None:
            raise ValueError("dimension should not be specified for sparse indexes")

        args = parse_non_empty_args(
            [
                ("name", name),
                ("dimension", dimension),
                ("metric", metric),
                ("spec", index_spec),
                ("deletion_protection", dp),
                ("vector_type", vector_type),
                ("tags", tags_obj),
            ]
        )

        return CreateIndexRequest(**args)

    @staticmethod
    def create_index_for_model_request(
        name: str,
        cloud: Union[CloudProvider, str],
        region: Union[AwsRegion, GcpRegion, AzureRegion, str],
        embed: Union[IndexEmbed, CreateIndexForModelEmbedTypedDict],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
    ) -> CreateIndexForModelRequest:
        cloud = convert_enum_to_string(cloud)
        region = convert_enum_to_string(region)
        if deletion_protection is not None:
            dp = PineconeDBControlRequestFactory.__parse_deletion_protection(deletion_protection)
        else:
            dp = None
        tags_obj = PineconeDBControlRequestFactory.__parse_tags(tags)

        if isinstance(embed, IndexEmbed):
            parsed_embed = embed.as_dict()
        else:
            # if dict, we need to parse enum values, if any, to string
            # and verify required fields are present
            required_fields = ["model", "field_map"]
            for field in required_fields:
                if field not in embed:
                    raise ValueError(f"{field} is required in embed")
            parsed_embed = {}
            for key, value in embed.items():
                if isinstance(value, Enum):
                    parsed_embed[key] = convert_enum_to_string(value)
                else:
                    parsed_embed[key] = value

        args = parse_non_empty_args(
            [
                ("name", name),
                ("cloud", cloud),
                ("region", region),
                ("embed", CreateIndexForModelRequestEmbed(**parsed_embed)),
                ("deletion_protection", dp),
                ("tags", tags_obj),
            ]
        )

        return CreateIndexForModelRequest(**args)

    @staticmethod
    def create_index_from_backup_request(
        name: str,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        tags: Optional[Dict[str, str]] = None,
    ) -> CreateIndexFromBackupRequest:
        if deletion_protection is not None:
            dp = PineconeDBControlRequestFactory.__parse_deletion_protection(deletion_protection)
        else:
            dp = None

        tags_obj = PineconeDBControlRequestFactory.__parse_tags(tags)

        return CreateIndexFromBackupRequest(name=name, deletion_protection=dp, tags=tags_obj)

    @staticmethod
    def configure_index_request(
        description: IndexModel,
        replicas: Optional[int] = None,
        pod_type: Optional[Union[PodType, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        if deletion_protection is None:
            dp = DeletionProtectionModel(description.deletion_protection)
        elif isinstance(deletion_protection, DeletionProtection):
            dp = DeletionProtectionModel(deletion_protection.value)
        elif deletion_protection in ["enabled", "disabled"]:
            dp = DeletionProtectionModel(deletion_protection)
        else:
            raise ValueError("deletion_protection must be either 'enabled' or 'disabled'")

        fetched_tags = description.tags
        if fetched_tags is None:
            starting_tags = {}
        else:
            starting_tags = fetched_tags.to_dict()

        if tags is None:
            # Do not modify tags if none are provided
            tags = starting_tags
        else:
            # Merge existing tags with new tags
            tags = {**starting_tags, **tags}

        pod_config_args: Dict[str, Any] = {}
        if pod_type:
            new_pod_type = convert_enum_to_string(pod_type)
            pod_config_args.update(pod_type=new_pod_type)
        if replicas:
            pod_config_args.update(replicas=replicas)

        if pod_config_args != {}:
            spec = ConfigureIndexRequestSpec(pod=ConfigureIndexRequestSpecPod(**pod_config_args))
            req = ConfigureIndexRequest(deletion_protection=dp, spec=spec, tags=IndexTags(**tags))
        else:
            req = ConfigureIndexRequest(deletion_protection=dp, tags=IndexTags(**tags))

        return req

    @staticmethod
    def create_collection_request(name: str, source: str) -> CreateCollectionRequest:
        return CreateCollectionRequest(name=name, source=source)
