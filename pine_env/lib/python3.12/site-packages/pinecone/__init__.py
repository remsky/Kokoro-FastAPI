"""
.. include:: ../pdoc/README.md
"""

from .deprecated_plugins import check_for_deprecated_plugins as _check_for_deprecated_plugins
from .deprecation_warnings import *
from .pinecone import Pinecone
from .pinecone_asyncio import PineconeAsyncio
from .exceptions import (
    PineconeException,
    PineconeApiTypeError,
    PineconeApiValueError,
    PineconeApiAttributeError,
    PineconeApiKeyError,
    PineconeApiException,
    NotFoundException,
    UnauthorizedException,
    ForbiddenException,
    ServiceException,
    PineconeProtocolError,
    PineconeConfigurationError,
    ListConversionException,
)

from .utils import __version__

import logging

# Set up lazy import handling
from .utils.lazy_imports import setup_lazy_imports as _setup_lazy_imports

_inference_lazy_imports = {
    "RerankModel": ("pinecone.inference", "RerankModel"),
    "EmbedModel": ("pinecone.inference", "EmbedModel"),
    "ModelInfo": ("pinecone.inference.models", "ModelInfo"),
    "ModelInfoList": ("pinecone.inference.models", "ModelInfoList"),
    "EmbeddingsList": ("pinecone.inference.models", "EmbeddingsList"),
    "RerankResult": ("pinecone.inference.models", "RerankResult"),
}

_db_data_lazy_imports = {
    "Vector": ("pinecone.db_data.dataclasses", "Vector"),
    "SparseValues": ("pinecone.db_data.dataclasses", "SparseValues"),
    "SearchQuery": ("pinecone.db_data.dataclasses", "SearchQuery"),
    "SearchQueryVector": ("pinecone.db_data.dataclasses", "SearchQueryVector"),
    "SearchRerank": ("pinecone.db_data.dataclasses", "SearchRerank"),
    "FetchResponse": ("pinecone.db_data.dataclasses", "FetchResponse"),
    "DeleteRequest": ("pinecone.db_data.models", "DeleteRequest"),
    "DescribeIndexStatsRequest": ("pinecone.db_data.models", "DescribeIndexStatsRequest"),
    "DescribeIndexStatsResponse": ("pinecone.db_data.models", "IndexDescription"),
    "RpcStatus": ("pinecone.db_data.models", "RpcStatus"),
    "ScoredVector": ("pinecone.db_data.models", "ScoredVector"),
    "SingleQueryResults": ("pinecone.db_data.models", "SingleQueryResults"),
    "QueryRequest": ("pinecone.db_data.models", "QueryRequest"),
    "QueryResponse": ("pinecone.db_data.models", "QueryResponse"),
    "UpsertResponse": ("pinecone.db_data.models", "UpsertResponse"),
    "UpdateRequest": ("pinecone.db_data.models", "UpdateRequest"),
    "ImportErrorMode": ("pinecone.db_data.resources.sync.bulk_import", "ImportErrorMode"),
    "VectorDictionaryMissingKeysError": (
        "pinecone.db_data.errors",
        "VectorDictionaryMissingKeysError",
    ),
    "VectorDictionaryExcessKeysError": (
        "pinecone.db_data.errors",
        "VectorDictionaryExcessKeysError",
    ),
    "VectorTupleLengthError": ("pinecone.db_data.errors", "VectorTupleLengthError"),
    "SparseValuesTypeError": ("pinecone.db_data.errors", "SparseValuesTypeError"),
    "SparseValuesMissingKeysError": ("pinecone.db_data.errors", "SparseValuesMissingKeysError"),
    "SparseValuesDictionaryExpectedError": (
        "pinecone.db_data.errors",
        "SparseValuesDictionaryExpectedError",
    ),
    "Index": ("pinecone.db_data.import_error", "Index"),
    "Inference": ("pinecone.db_data.import_error", "Inference"),
}

_db_control_lazy_imports = {
    "CloudProvider": ("pinecone.db_control.enums", "CloudProvider"),
    "AwsRegion": ("pinecone.db_control.enums", "AwsRegion"),
    "GcpRegion": ("pinecone.db_control.enums", "GcpRegion"),
    "AzureRegion": ("pinecone.db_control.enums", "AzureRegion"),
    "PodIndexEnvironment": ("pinecone.db_control.enums", "PodIndexEnvironment"),
    "Metric": ("pinecone.db_control.enums", "Metric"),
    "VectorType": ("pinecone.db_control.enums", "VectorType"),
    "DeletionProtection": ("pinecone.db_control.enums", "DeletionProtection"),
    "CollectionDescription": ("pinecone.db_control.models", "CollectionDescription"),
    "CollectionList": ("pinecone.db_control.models", "CollectionList"),
    "IndexList": ("pinecone.db_control.models", "IndexList"),
    "IndexModel": ("pinecone.db_control.models", "IndexModel"),
    "IndexEmbed": ("pinecone.db_control.models", "IndexEmbed"),
    "ByocSpec": ("pinecone.db_control.models", "ByocSpec"),
    "ServerlessSpec": ("pinecone.db_control.models", "ServerlessSpec"),
    "ServerlessSpecDefinition": ("pinecone.db_control.models", "ServerlessSpecDefinition"),
    "PodSpec": ("pinecone.db_control.models", "PodSpec"),
    "PodSpecDefinition": ("pinecone.db_control.models", "PodSpecDefinition"),
    "PodType": ("pinecone.db_control.enums", "PodType"),
    "RestoreJobModel": ("pinecone.db_control.models", "RestoreJobModel"),
    "RestoreJobList": ("pinecone.db_control.models", "RestoreJobList"),
    "BackupModel": ("pinecone.db_control.models", "BackupModel"),
    "BackupList": ("pinecone.db_control.models", "BackupList"),
}

_config_lazy_imports = {
    "Config": ("pinecone.config", "Config"),
    "ConfigBuilder": ("pinecone.config", "ConfigBuilder"),
    "PineconeConfig": ("pinecone.config", "PineconeConfig"),
}

# Define imports to be lazily loaded
_LAZY_IMPORTS = {
    **_inference_lazy_imports,
    **_db_data_lazy_imports,
    **_db_control_lazy_imports,
    **_config_lazy_imports,
}

# Set up the lazy import handler
_setup_lazy_imports(_LAZY_IMPORTS)

# Raise an exception if the user is attempting to use the SDK with
# deprecated plugins installed in their project.
_check_for_deprecated_plugins()

# Silence annoying log messages from the plugin interface
logging.getLogger("pinecone_plugin_interface").setLevel(logging.CRITICAL)

__all__ = [
    "__version__",
    # Deprecated top-levelfunctions
    "init",
    "create_index",
    "delete_index",
    "list_indexes",
    "describe_index",
    "configure_index",
    "scale_index",
    "create_collection",
    "delete_collection",
    "describe_collection",
    "list_collections",
    # Primary client classes
    "Pinecone",
    "PineconeAsyncio",
    # All lazy-loaded types
    *list(_LAZY_IMPORTS.keys()),
    # Exception classes
    "PineconeException",
    "PineconeApiException",
    "PineconeConfigurationError",
    "PineconeProtocolError",
    "PineconeApiAttributeError",
    "PineconeApiTypeError",
    "PineconeApiValueError",
    "PineconeApiKeyError",
    "NotFoundException",
    "UnauthorizedException",
    "ForbiddenException",
    "ServiceException",
    "ListConversionException",
    "VectorDictionaryMissingKeysError",
    "VectorDictionaryExcessKeysError",
    "VectorTupleLengthError",
    "SparseValuesTypeError",
    "SparseValuesMissingKeysError",
    "SparseValuesDictionaryExpectedError",
]
