"""
Pinecone Control Plane API

Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: 2025-04
Contact: support@pinecone.io
"""

from pinecone.openapi_support import ApiClient, AsyncioApiClient
from pinecone.openapi_support.endpoint_utils import (
    ExtraOpenApiKwargsTypedDict,
    KwargsWithOpenApiKwargDefaultsTypedDict,
)
from pinecone.openapi_support.endpoint import Endpoint as _Endpoint, ExtraOpenApiKwargsTypedDict
from pinecone.openapi_support.asyncio_endpoint import AsyncioEndpoint as _AsyncioEndpoint
from pinecone.openapi_support.model_utils import (  # noqa: F401
    date,
    datetime,
    file_type,
    none_type,
    validate_and_convert_types,
)
from pinecone.core.openapi.db_control.model.backup_list import BackupList
from pinecone.core.openapi.db_control.model.backup_model import BackupModel
from pinecone.core.openapi.db_control.model.collection_list import CollectionList
from pinecone.core.openapi.db_control.model.collection_model import CollectionModel
from pinecone.core.openapi.db_control.model.configure_index_request import ConfigureIndexRequest
from pinecone.core.openapi.db_control.model.create_backup_request import CreateBackupRequest
from pinecone.core.openapi.db_control.model.create_collection_request import CreateCollectionRequest
from pinecone.core.openapi.db_control.model.create_index_for_model_request import (
    CreateIndexForModelRequest,
)
from pinecone.core.openapi.db_control.model.create_index_from_backup_request import (
    CreateIndexFromBackupRequest,
)
from pinecone.core.openapi.db_control.model.create_index_from_backup_response import (
    CreateIndexFromBackupResponse,
)
from pinecone.core.openapi.db_control.model.create_index_request import CreateIndexRequest
from pinecone.core.openapi.db_control.model.error_response import ErrorResponse
from pinecone.core.openapi.db_control.model.index_list import IndexList
from pinecone.core.openapi.db_control.model.index_model import IndexModel
from pinecone.core.openapi.db_control.model.restore_job_list import RestoreJobList
from pinecone.core.openapi.db_control.model.restore_job_model import RestoreJobModel


class ManageIndexesApi:
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

        def __configure_index(
            self, index_name, configure_index_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Configure an index  # noqa: E501

            Configure an existing index. For serverless indexes, you can configure index deletion protection, tags, and integrated inference embedding settings for the index. For pod-based indexes, you can configure the pod size, number of replicas, tags, and index deletion protection.  It is not possible to change the pod type of a pod-based index. However, you can create a collection from a pod-based index and then [create a new pod-based index with a different pod type](http://docs.pinecone.io/guides/indexes/pods/create-a-pod-based-index#create-a-pod-index-from-a-collection) from the collection. For guidance and examples, see [Configure an index](http://docs.pinecone.io/guides/indexes/pods/manage-pod-based-indexes).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.configure_index(index_name, configure_index_request, async_req=True)
            >>> result = thread.get()

            Args:
                index_name (str): The name of the index to configure.
                configure_index_request (ConfigureIndexRequest): The desired pod size and replica configuration for the index.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            kwargs["configure_index_request"] = configure_index_request
            return self.call_with_http_info(**kwargs)

        self.configure_index = _Endpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "configure_index",
                "http_method": "PATCH",
                "servers": None,
            },
            params_map={
                "all": ["index_name", "configure_index_request"],
                "required": ["index_name", "configure_index_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "index_name": (str,),
                    "configure_index_request": (ConfigureIndexRequest,),
                },
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path", "configure_index_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__configure_index,
        )

        def __create_backup(
            self, index_name, create_backup_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Create a backup of an index  # noqa: E501

            Create a backup of an index.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_backup(index_name, create_backup_request, async_req=True)
            >>> result = thread.get()

            Args:
                index_name (str): Name of the index to backup
                create_backup_request (CreateBackupRequest): The desired configuration for the backup.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                BackupModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            kwargs["create_backup_request"] = create_backup_request
            return self.call_with_http_info(**kwargs)

        self.create_backup = _Endpoint(
            settings={
                "response_type": (BackupModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}/backups",
                "operation_id": "create_backup",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["index_name", "create_backup_request"],
                "required": ["index_name", "create_backup_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "index_name": (str,),
                    "create_backup_request": (CreateBackupRequest,),
                },
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path", "create_backup_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_backup,
        )

        def __create_collection(
            self, create_collection_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Create a collection  # noqa: E501

            Create a Pinecone collection.    Serverless indexes do not support collections.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_collection(create_collection_request, async_req=True)
            >>> result = thread.get()

            Args:
                create_collection_request (CreateCollectionRequest): The desired configuration for the collection.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                CollectionModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["create_collection_request"] = create_collection_request
            return self.call_with_http_info(**kwargs)

        self.create_collection = _Endpoint(
            settings={
                "response_type": (CollectionModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections",
                "operation_id": "create_collection",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_collection_request"],
                "required": ["create_collection_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_collection_request": (CreateCollectionRequest,)},
                "attribute_map": {},
                "location_map": {"create_collection_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_collection,
        )

        def __create_index(self, create_index_request, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Create an index  # noqa: E501

            Create a Pinecone index. This is where you specify the measure of similarity, the dimension of vectors to be stored in the index, which cloud provider you would like to deploy with, and more.    For guidance and examples, see [Create an index](https://docs.pinecone.io/guides/index-data/create-an-index).   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_index(create_index_request, async_req=True)
            >>> result = thread.get()

            Args:
                create_index_request (CreateIndexRequest): The desired configuration for the index.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["create_index_request"] = create_index_request
            return self.call_with_http_info(**kwargs)

        self.create_index = _Endpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes",
                "operation_id": "create_index",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_index_request"],
                "required": ["create_index_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_index_request": (CreateIndexRequest,)},
                "attribute_map": {},
                "location_map": {"create_index_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index,
        )

        def __create_index_for_model(
            self, create_index_for_model_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Create an index with integrated embedding  # noqa: E501

            Create an index with integrated embedding.  With this type of index, you provide source text, and Pinecone uses a [hosted embedding model](https://docs.pinecone.io/guides/index-data/create-an-index#embedding-models) to convert the text automatically during [upsert](https://docs.pinecone.io/reference/api/2025-01/data-plane/upsert_records) and [search](https://docs.pinecone.io/reference/api/2025-01/data-plane/search_records).  For guidance and examples, see [Create an index](https://docs.pinecone.io/guides/index-data/create-an-index#integrated-embedding).  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_index_for_model(create_index_for_model_request, async_req=True)
            >>> result = thread.get()

            Args:
                create_index_for_model_request (CreateIndexForModelRequest): The desired configuration for the index and associated embedding model.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["create_index_for_model_request"] = create_index_for_model_request
            return self.call_with_http_info(**kwargs)

        self.create_index_for_model = _Endpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/create-for-model",
                "operation_id": "create_index_for_model",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_index_for_model_request"],
                "required": ["create_index_for_model_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_index_for_model_request": (CreateIndexForModelRequest,)},
                "attribute_map": {},
                "location_map": {"create_index_for_model_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index_for_model,
        )

        def __create_index_from_backup_operation(
            self, backup_id, create_index_from_backup_request, **kwargs: ExtraOpenApiKwargsTypedDict
        ):
            """Create an index from a backup  # noqa: E501

            Create an index from a backup.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.create_index_from_backup_operation(backup_id, create_index_from_backup_request, async_req=True)
            >>> result = thread.get()

            Args:
                backup_id (str): The ID of the backup to create an index from.
                create_index_from_backup_request (CreateIndexFromBackupRequest): The desired configuration for the index created from a backup.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                CreateIndexFromBackupResponse
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["backup_id"] = backup_id
            kwargs["create_index_from_backup_request"] = create_index_from_backup_request
            return self.call_with_http_info(**kwargs)

        self.create_index_from_backup_operation = _Endpoint(
            settings={
                "response_type": (CreateIndexFromBackupResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/backups/{backup_id}/create-index",
                "operation_id": "create_index_from_backup_operation",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["backup_id", "create_index_from_backup_request"],
                "required": ["backup_id", "create_index_from_backup_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "backup_id": (str,),
                    "create_index_from_backup_request": (CreateIndexFromBackupRequest,),
                },
                "attribute_map": {"backup_id": "backup_id"},
                "location_map": {"backup_id": "path", "create_index_from_backup_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index_from_backup_operation,
        )

        def __delete_backup(self, backup_id, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Delete a backup  # noqa: E501

            Delete a backup.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_backup(backup_id, async_req=True)
            >>> result = thread.get()

            Args:
                backup_id (str): The ID of the backup to delete.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                None
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["backup_id"] = backup_id
            return self.call_with_http_info(**kwargs)

        self.delete_backup = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/backups/{backup_id}",
                "operation_id": "delete_backup",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["backup_id"],
                "required": ["backup_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"backup_id": (str,)},
                "attribute_map": {"backup_id": "backup_id"},
                "location_map": {"backup_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_backup,
        )

        def __delete_collection(self, collection_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Delete a collection  # noqa: E501

            Delete an existing collection. Serverless indexes do not support collections.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_collection(collection_name, async_req=True)
            >>> result = thread.get()

            Args:
                collection_name (str): The name of the collection.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                None
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["collection_name"] = collection_name
            return self.call_with_http_info(**kwargs)

        self.delete_collection = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections/{collection_name}",
                "operation_id": "delete_collection",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["collection_name"],
                "required": ["collection_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"collection_name": (str,)},
                "attribute_map": {"collection_name": "collection_name"},
                "location_map": {"collection_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_collection,
        )

        def __delete_index(self, index_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Delete an index  # noqa: E501

            Delete an existing index.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.delete_index(index_name, async_req=True)
            >>> result = thread.get()

            Args:
                index_name (str): The name of the index to delete.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                None
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            return self.call_with_http_info(**kwargs)

        self.delete_index = _Endpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "delete_index",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["index_name"],
                "required": ["index_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"index_name": (str,)},
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_index,
        )

        def __describe_backup(self, backup_id, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Describe a backup  # noqa: E501

            Get a description of a backup.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_backup(backup_id, async_req=True)
            >>> result = thread.get()

            Args:
                backup_id (str): The ID of the backup to describe.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                BackupModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["backup_id"] = backup_id
            return self.call_with_http_info(**kwargs)

        self.describe_backup = _Endpoint(
            settings={
                "response_type": (BackupModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/backups/{backup_id}",
                "operation_id": "describe_backup",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["backup_id"],
                "required": ["backup_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"backup_id": (str,)},
                "attribute_map": {"backup_id": "backup_id"},
                "location_map": {"backup_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_backup,
        )

        def __describe_collection(self, collection_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Describe a collection  # noqa: E501

            Get a description of a collection. Serverless indexes do not support collections.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_collection(collection_name, async_req=True)
            >>> result = thread.get()

            Args:
                collection_name (str): The name of the collection to be described.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                CollectionModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["collection_name"] = collection_name
            return self.call_with_http_info(**kwargs)

        self.describe_collection = _Endpoint(
            settings={
                "response_type": (CollectionModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections/{collection_name}",
                "operation_id": "describe_collection",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["collection_name"],
                "required": ["collection_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"collection_name": (str,)},
                "attribute_map": {"collection_name": "collection_name"},
                "location_map": {"collection_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_collection,
        )

        def __describe_index(self, index_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Describe an index  # noqa: E501

            Get a description of an index.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_index(index_name, async_req=True)
            >>> result = thread.get()

            Args:
                index_name (str): The name of the index to be described.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                IndexModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            return self.call_with_http_info(**kwargs)

        self.describe_index = _Endpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "describe_index",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["index_name"],
                "required": ["index_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"index_name": (str,)},
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_index,
        )

        def __describe_restore_job(self, job_id, **kwargs: ExtraOpenApiKwargsTypedDict):
            """Describe a restore job  # noqa: E501

            Get a description of a restore job.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.describe_restore_job(job_id, async_req=True)
            >>> result = thread.get()

            Args:
                job_id (str): The ID of the restore job to describe.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                RestoreJobModel
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["job_id"] = job_id
            return self.call_with_http_info(**kwargs)

        self.describe_restore_job = _Endpoint(
            settings={
                "response_type": (RestoreJobModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/restore-jobs/{job_id}",
                "operation_id": "describe_restore_job",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["job_id"],
                "required": ["job_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"job_id": (str,)},
                "attribute_map": {"job_id": "job_id"},
                "location_map": {"job_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_restore_job,
        )

        def __list_collections(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List collections  # noqa: E501

            List all collections in a project. Serverless indexes do not support collections.   # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_collections(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                CollectionList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_collections = _Endpoint(
            settings={
                "response_type": (CollectionList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections",
                "operation_id": "list_collections",
                "http_method": "GET",
                "servers": None,
            },
            params_map={"all": [], "required": [], "nullable": [], "enum": [], "validation": []},
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {},
                "attribute_map": {},
                "location_map": {},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_collections,
        )

        def __list_index_backups(self, index_name, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List backups for an index  # noqa: E501

            List all backups for an index.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_index_backups(index_name, async_req=True)
            >>> result = thread.get()

            Args:
                index_name (str): Name of the backed up index

            Keyword Args:
                limit (int): The number of results to return per page. [optional] if omitted the server will use the default value of 10.
                pagination_token (str): The token to use to retrieve the next page of results. [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                BackupList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            return self.call_with_http_info(**kwargs)

        self.list_index_backups = _Endpoint(
            settings={
                "response_type": (BackupList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}/backups",
                "operation_id": "list_index_backups",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["index_name", "limit", "pagination_token"],
                "required": ["index_name"],
                "nullable": [],
                "enum": [],
                "validation": ["limit"],
            },
            root_map={
                "validations": {("limit",): {"inclusive_maximum": 100, "inclusive_minimum": 1}},
                "allowed_values": {},
                "openapi_types": {
                    "index_name": (str,),
                    "limit": (int,),
                    "pagination_token": (str,),
                },
                "attribute_map": {
                    "index_name": "index_name",
                    "limit": "limit",
                    "pagination_token": "paginationToken",
                },
                "location_map": {
                    "index_name": "path",
                    "limit": "query",
                    "pagination_token": "query",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_index_backups,
        )

        def __list_indexes(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List indexes  # noqa: E501

            List all indexes in a project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_indexes(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                IndexList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_indexes = _Endpoint(
            settings={
                "response_type": (IndexList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes",
                "operation_id": "list_indexes",
                "http_method": "GET",
                "servers": None,
            },
            params_map={"all": [], "required": [], "nullable": [], "enum": [], "validation": []},
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {},
                "attribute_map": {},
                "location_map": {},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_indexes,
        )

        def __list_project_backups(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List backups for all indexes in a project  # noqa: E501

            List all backups for a project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_project_backups(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                limit (int): The number of results to return per page. [optional] if omitted the server will use the default value of 10.
                pagination_token (str): The token to use to retrieve the next page of results. [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                BackupList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_project_backups = _Endpoint(
            settings={
                "response_type": (BackupList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/backups",
                "operation_id": "list_project_backups",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["limit", "pagination_token"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": ["limit"],
            },
            root_map={
                "validations": {("limit",): {"inclusive_maximum": 100, "inclusive_minimum": 1}},
                "allowed_values": {},
                "openapi_types": {"limit": (int,), "pagination_token": (str,)},
                "attribute_map": {"limit": "limit", "pagination_token": "paginationToken"},
                "location_map": {"limit": "query", "pagination_token": "query"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_project_backups,
        )

        def __list_restore_jobs(self, **kwargs: ExtraOpenApiKwargsTypedDict):
            """List restore jobs  # noqa: E501

            List all restore jobs for a project.  # noqa: E501
            This method makes a synchronous HTTP request by default. To make an
            asynchronous HTTP request, please pass async_req=True

            >>> thread = api.list_restore_jobs(async_req=True)
            >>> result = thread.get()


            Keyword Args:
                limit (int): The number of results to return per page. [optional] if omitted the server will use the default value of 10.
                pagination_token (str): The token to use to retrieve the next page of results. [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.
                async_req (bool): execute request asynchronously

            Returns:
                RestoreJobList
                    If the method is called asynchronously, returns the request
                    thread.
            """
            kwargs = self._process_openapi_kwargs(kwargs)
            return self.call_with_http_info(**kwargs)

        self.list_restore_jobs = _Endpoint(
            settings={
                "response_type": (RestoreJobList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/restore-jobs",
                "operation_id": "list_restore_jobs",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["limit", "pagination_token"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": ["limit"],
            },
            root_map={
                "validations": {("limit",): {"inclusive_maximum": 100, "inclusive_minimum": 1}},
                "allowed_values": {},
                "openapi_types": {"limit": (int,), "pagination_token": (str,)},
                "attribute_map": {"limit": "limit", "pagination_token": "paginationToken"},
                "location_map": {"limit": "query", "pagination_token": "query"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_restore_jobs,
        )


class AsyncioManageIndexesApi:
    """NOTE: This class is @generated using OpenAPI

    Do not edit the class manually.
    """

    def __init__(self, api_client=None) -> None:
        if api_client is None:
            api_client = AsyncioApiClient()
        self.api_client = api_client

        async def __configure_index(self, index_name, configure_index_request, **kwargs):
            """Configure an index  # noqa: E501

            Configure an existing index. For serverless indexes, you can configure index deletion protection, tags, and integrated inference embedding settings for the index. For pod-based indexes, you can configure the pod size, number of replicas, tags, and index deletion protection.  It is not possible to change the pod type of a pod-based index. However, you can create a collection from a pod-based index and then [create a new pod-based index with a different pod type](http://docs.pinecone.io/guides/indexes/pods/create-a-pod-based-index#create-a-pod-index-from-a-collection) from the collection. For guidance and examples, see [Configure an index](http://docs.pinecone.io/guides/indexes/pods/manage-pod-based-indexes).  # noqa: E501


            Args:
                index_name (str): The name of the index to configure.
                configure_index_request (ConfigureIndexRequest): The desired pod size and replica configuration for the index.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                IndexModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            kwargs["configure_index_request"] = configure_index_request
            return await self.call_with_http_info(**kwargs)

        self.configure_index = _AsyncioEndpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "configure_index",
                "http_method": "PATCH",
                "servers": None,
            },
            params_map={
                "all": ["index_name", "configure_index_request"],
                "required": ["index_name", "configure_index_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "index_name": (str,),
                    "configure_index_request": (ConfigureIndexRequest,),
                },
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path", "configure_index_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__configure_index,
        )

        async def __create_backup(self, index_name, create_backup_request, **kwargs):
            """Create a backup of an index  # noqa: E501

            Create a backup of an index.   # noqa: E501


            Args:
                index_name (str): Name of the index to backup
                create_backup_request (CreateBackupRequest): The desired configuration for the backup.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                BackupModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            kwargs["create_backup_request"] = create_backup_request
            return await self.call_with_http_info(**kwargs)

        self.create_backup = _AsyncioEndpoint(
            settings={
                "response_type": (BackupModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}/backups",
                "operation_id": "create_backup",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["index_name", "create_backup_request"],
                "required": ["index_name", "create_backup_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "index_name": (str,),
                    "create_backup_request": (CreateBackupRequest,),
                },
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path", "create_backup_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_backup,
        )

        async def __create_collection(self, create_collection_request, **kwargs):
            """Create a collection  # noqa: E501

            Create a Pinecone collection.    Serverless indexes do not support collections.   # noqa: E501


            Args:
                create_collection_request (CreateCollectionRequest): The desired configuration for the collection.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                CollectionModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["create_collection_request"] = create_collection_request
            return await self.call_with_http_info(**kwargs)

        self.create_collection = _AsyncioEndpoint(
            settings={
                "response_type": (CollectionModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections",
                "operation_id": "create_collection",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_collection_request"],
                "required": ["create_collection_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_collection_request": (CreateCollectionRequest,)},
                "attribute_map": {},
                "location_map": {"create_collection_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_collection,
        )

        async def __create_index(self, create_index_request, **kwargs):
            """Create an index  # noqa: E501

            Create a Pinecone index. This is where you specify the measure of similarity, the dimension of vectors to be stored in the index, which cloud provider you would like to deploy with, and more.    For guidance and examples, see [Create an index](https://docs.pinecone.io/guides/index-data/create-an-index).   # noqa: E501


            Args:
                create_index_request (CreateIndexRequest): The desired configuration for the index.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                IndexModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["create_index_request"] = create_index_request
            return await self.call_with_http_info(**kwargs)

        self.create_index = _AsyncioEndpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes",
                "operation_id": "create_index",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_index_request"],
                "required": ["create_index_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_index_request": (CreateIndexRequest,)},
                "attribute_map": {},
                "location_map": {"create_index_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index,
        )

        async def __create_index_for_model(self, create_index_for_model_request, **kwargs):
            """Create an index with integrated embedding  # noqa: E501

            Create an index with integrated embedding.  With this type of index, you provide source text, and Pinecone uses a [hosted embedding model](https://docs.pinecone.io/guides/index-data/create-an-index#embedding-models) to convert the text automatically during [upsert](https://docs.pinecone.io/reference/api/2025-01/data-plane/upsert_records) and [search](https://docs.pinecone.io/reference/api/2025-01/data-plane/search_records).  For guidance and examples, see [Create an index](https://docs.pinecone.io/guides/index-data/create-an-index#integrated-embedding).  # noqa: E501


            Args:
                create_index_for_model_request (CreateIndexForModelRequest): The desired configuration for the index and associated embedding model.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                IndexModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["create_index_for_model_request"] = create_index_for_model_request
            return await self.call_with_http_info(**kwargs)

        self.create_index_for_model = _AsyncioEndpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/create-for-model",
                "operation_id": "create_index_for_model",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["create_index_for_model_request"],
                "required": ["create_index_for_model_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"create_index_for_model_request": (CreateIndexForModelRequest,)},
                "attribute_map": {},
                "location_map": {"create_index_for_model_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index_for_model,
        )

        async def __create_index_from_backup_operation(
            self, backup_id, create_index_from_backup_request, **kwargs
        ):
            """Create an index from a backup  # noqa: E501

            Create an index from a backup.  # noqa: E501


            Args:
                backup_id (str): The ID of the backup to create an index from.
                create_index_from_backup_request (CreateIndexFromBackupRequest): The desired configuration for the index created from a backup.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                CreateIndexFromBackupResponse
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["backup_id"] = backup_id
            kwargs["create_index_from_backup_request"] = create_index_from_backup_request
            return await self.call_with_http_info(**kwargs)

        self.create_index_from_backup_operation = _AsyncioEndpoint(
            settings={
                "response_type": (CreateIndexFromBackupResponse,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/backups/{backup_id}/create-index",
                "operation_id": "create_index_from_backup_operation",
                "http_method": "POST",
                "servers": None,
            },
            params_map={
                "all": ["backup_id", "create_index_from_backup_request"],
                "required": ["backup_id", "create_index_from_backup_request"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {
                    "backup_id": (str,),
                    "create_index_from_backup_request": (CreateIndexFromBackupRequest,),
                },
                "attribute_map": {"backup_id": "backup_id"},
                "location_map": {"backup_id": "path", "create_index_from_backup_request": "body"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": ["application/json"]},
            api_client=api_client,
            callable=__create_index_from_backup_operation,
        )

        async def __delete_backup(self, backup_id, **kwargs):
            """Delete a backup  # noqa: E501

            Delete a backup.  # noqa: E501


            Args:
                backup_id (str): The ID of the backup to delete.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                None
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["backup_id"] = backup_id
            return await self.call_with_http_info(**kwargs)

        self.delete_backup = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/backups/{backup_id}",
                "operation_id": "delete_backup",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["backup_id"],
                "required": ["backup_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"backup_id": (str,)},
                "attribute_map": {"backup_id": "backup_id"},
                "location_map": {"backup_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_backup,
        )

        async def __delete_collection(self, collection_name, **kwargs):
            """Delete a collection  # noqa: E501

            Delete an existing collection. Serverless indexes do not support collections.   # noqa: E501


            Args:
                collection_name (str): The name of the collection.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                None
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["collection_name"] = collection_name
            return await self.call_with_http_info(**kwargs)

        self.delete_collection = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections/{collection_name}",
                "operation_id": "delete_collection",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["collection_name"],
                "required": ["collection_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"collection_name": (str,)},
                "attribute_map": {"collection_name": "collection_name"},
                "location_map": {"collection_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_collection,
        )

        async def __delete_index(self, index_name, **kwargs):
            """Delete an index  # noqa: E501

            Delete an existing index.  # noqa: E501


            Args:
                index_name (str): The name of the index to delete.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                None
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            return await self.call_with_http_info(**kwargs)

        self.delete_index = _AsyncioEndpoint(
            settings={
                "response_type": None,
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "delete_index",
                "http_method": "DELETE",
                "servers": None,
            },
            params_map={
                "all": ["index_name"],
                "required": ["index_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"index_name": (str,)},
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__delete_index,
        )

        async def __describe_backup(self, backup_id, **kwargs):
            """Describe a backup  # noqa: E501

            Get a description of a backup.  # noqa: E501


            Args:
                backup_id (str): The ID of the backup to describe.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                BackupModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["backup_id"] = backup_id
            return await self.call_with_http_info(**kwargs)

        self.describe_backup = _AsyncioEndpoint(
            settings={
                "response_type": (BackupModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/backups/{backup_id}",
                "operation_id": "describe_backup",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["backup_id"],
                "required": ["backup_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"backup_id": (str,)},
                "attribute_map": {"backup_id": "backup_id"},
                "location_map": {"backup_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_backup,
        )

        async def __describe_collection(self, collection_name, **kwargs):
            """Describe a collection  # noqa: E501

            Get a description of a collection. Serverless indexes do not support collections.   # noqa: E501


            Args:
                collection_name (str): The name of the collection to be described.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                CollectionModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["collection_name"] = collection_name
            return await self.call_with_http_info(**kwargs)

        self.describe_collection = _AsyncioEndpoint(
            settings={
                "response_type": (CollectionModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections/{collection_name}",
                "operation_id": "describe_collection",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["collection_name"],
                "required": ["collection_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"collection_name": (str,)},
                "attribute_map": {"collection_name": "collection_name"},
                "location_map": {"collection_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_collection,
        )

        async def __describe_index(self, index_name, **kwargs):
            """Describe an index  # noqa: E501

            Get a description of an index.  # noqa: E501


            Args:
                index_name (str): The name of the index to be described.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                IndexModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            return await self.call_with_http_info(**kwargs)

        self.describe_index = _AsyncioEndpoint(
            settings={
                "response_type": (IndexModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}",
                "operation_id": "describe_index",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["index_name"],
                "required": ["index_name"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"index_name": (str,)},
                "attribute_map": {"index_name": "index_name"},
                "location_map": {"index_name": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_index,
        )

        async def __describe_restore_job(self, job_id, **kwargs):
            """Describe a restore job  # noqa: E501

            Get a description of a restore job.  # noqa: E501


            Args:
                job_id (str): The ID of the restore job to describe.

            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                RestoreJobModel
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["job_id"] = job_id
            return await self.call_with_http_info(**kwargs)

        self.describe_restore_job = _AsyncioEndpoint(
            settings={
                "response_type": (RestoreJobModel,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/restore-jobs/{job_id}",
                "operation_id": "describe_restore_job",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["job_id"],
                "required": ["job_id"],
                "nullable": [],
                "enum": [],
                "validation": [],
            },
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {"job_id": (str,)},
                "attribute_map": {"job_id": "job_id"},
                "location_map": {"job_id": "path"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__describe_restore_job,
        )

        async def __list_collections(self, **kwargs):
            """List collections  # noqa: E501

            List all collections in a project. Serverless indexes do not support collections.   # noqa: E501



            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                CollectionList
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_collections = _AsyncioEndpoint(
            settings={
                "response_type": (CollectionList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/collections",
                "operation_id": "list_collections",
                "http_method": "GET",
                "servers": None,
            },
            params_map={"all": [], "required": [], "nullable": [], "enum": [], "validation": []},
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {},
                "attribute_map": {},
                "location_map": {},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_collections,
        )

        async def __list_index_backups(self, index_name, **kwargs):
            """List backups for an index  # noqa: E501

            List all backups for an index.  # noqa: E501


            Args:
                index_name (str): Name of the backed up index

            Keyword Args:
                limit (int): The number of results to return per page. [optional] if omitted the server will use the default value of 10.
                pagination_token (str): The token to use to retrieve the next page of results. [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                BackupList
            """
            self._process_openapi_kwargs(kwargs)
            kwargs["index_name"] = index_name
            return await self.call_with_http_info(**kwargs)

        self.list_index_backups = _AsyncioEndpoint(
            settings={
                "response_type": (BackupList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes/{index_name}/backups",
                "operation_id": "list_index_backups",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["index_name", "limit", "pagination_token"],
                "required": ["index_name"],
                "nullable": [],
                "enum": [],
                "validation": ["limit"],
            },
            root_map={
                "validations": {("limit",): {"inclusive_maximum": 100, "inclusive_minimum": 1}},
                "allowed_values": {},
                "openapi_types": {
                    "index_name": (str,),
                    "limit": (int,),
                    "pagination_token": (str,),
                },
                "attribute_map": {
                    "index_name": "index_name",
                    "limit": "limit",
                    "pagination_token": "paginationToken",
                },
                "location_map": {
                    "index_name": "path",
                    "limit": "query",
                    "pagination_token": "query",
                },
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_index_backups,
        )

        async def __list_indexes(self, **kwargs):
            """List indexes  # noqa: E501

            List all indexes in a project.  # noqa: E501



            Keyword Args:
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                IndexList
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_indexes = _AsyncioEndpoint(
            settings={
                "response_type": (IndexList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/indexes",
                "operation_id": "list_indexes",
                "http_method": "GET",
                "servers": None,
            },
            params_map={"all": [], "required": [], "nullable": [], "enum": [], "validation": []},
            root_map={
                "validations": {},
                "allowed_values": {},
                "openapi_types": {},
                "attribute_map": {},
                "location_map": {},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_indexes,
        )

        async def __list_project_backups(self, **kwargs):
            """List backups for all indexes in a project  # noqa: E501

            List all backups for a project.  # noqa: E501



            Keyword Args:
                limit (int): The number of results to return per page. [optional] if omitted the server will use the default value of 10.
                pagination_token (str): The token to use to retrieve the next page of results. [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                BackupList
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_project_backups = _AsyncioEndpoint(
            settings={
                "response_type": (BackupList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/backups",
                "operation_id": "list_project_backups",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["limit", "pagination_token"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": ["limit"],
            },
            root_map={
                "validations": {("limit",): {"inclusive_maximum": 100, "inclusive_minimum": 1}},
                "allowed_values": {},
                "openapi_types": {"limit": (int,), "pagination_token": (str,)},
                "attribute_map": {"limit": "limit", "pagination_token": "paginationToken"},
                "location_map": {"limit": "query", "pagination_token": "query"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_project_backups,
        )

        async def __list_restore_jobs(self, **kwargs):
            """List restore jobs  # noqa: E501

            List all restore jobs for a project.  # noqa: E501



            Keyword Args:
                limit (int): The number of results to return per page. [optional] if omitted the server will use the default value of 10.
                pagination_token (str): The token to use to retrieve the next page of results. [optional]
                _return_http_data_only (bool): response data without head status
                    code and headers. Default is True.
                _preload_content (bool): if False, the urllib3.HTTPResponse object
                    will be returned without reading/decoding response data.
                    Default is True.
                _request_timeout (int/float/tuple): timeout setting for this request. If
                    one number provided, it will be total request timeout. It can also
                    be a pair (tuple) of (connection, read) timeouts.
                    Default is None.
                _check_input_type (bool): specifies if type checking
                    should be done one the data sent to the server.
                    Default is True.
                _check_return_type (bool): specifies if type checking
                    should be done one the data received from the server.
                    Default is True.

            Returns:
                RestoreJobList
            """
            self._process_openapi_kwargs(kwargs)
            return await self.call_with_http_info(**kwargs)

        self.list_restore_jobs = _AsyncioEndpoint(
            settings={
                "response_type": (RestoreJobList,),
                "auth": ["ApiKeyAuth"],
                "endpoint_path": "/restore-jobs",
                "operation_id": "list_restore_jobs",
                "http_method": "GET",
                "servers": None,
            },
            params_map={
                "all": ["limit", "pagination_token"],
                "required": [],
                "nullable": [],
                "enum": [],
                "validation": ["limit"],
            },
            root_map={
                "validations": {("limit",): {"inclusive_maximum": 100, "inclusive_minimum": 1}},
                "allowed_values": {},
                "openapi_types": {"limit": (int,), "pagination_token": (str,)},
                "attribute_map": {"limit": "limit", "pagination_token": "paginationToken"},
                "location_map": {"limit": "query", "pagination_token": "query"},
                "collection_format_map": {},
            },
            headers_map={"accept": ["application/json"], "content_type": []},
            api_client=api_client,
            callable=__list_restore_jobs,
        )
