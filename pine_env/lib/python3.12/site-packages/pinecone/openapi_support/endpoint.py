from .model_utils import none_type
from typing import Dict, List, Callable
from .api_client import ApiClient
from .api_client_utils import HeaderUtil
from .endpoint_utils import (
    EndpointUtils,
    EndpointParamsMapDict,
    EndpointRootMapDict,
    EndpointSettingsDict,
    ExtraOpenApiKwargsTypedDict,
)


class Endpoint:
    def __init__(
        self,
        settings: EndpointSettingsDict,
        params_map: EndpointParamsMapDict,
        root_map: EndpointRootMapDict,
        headers_map: Dict[str, List[str]],
        api_client: ApiClient,
        callable: Callable,
    ):
        """Creates an endpoint

        Args:
            settings (dict): see below key value pairs
                'response_type' (tuple/None): response type
                'auth' (list): a list of auth type keys
                'endpoint_path' (str): the endpoint path
                'operation_id' (str): endpoint string identifier
                'http_method' (str): POST/PUT/PATCH/GET etc
                'servers' (list): list of str servers that this endpoint is at
            params_map (dict): see below key value pairs
                'all' (list): list of str endpoint parameter names
                'required' (list): list of required parameter names
                'nullable' (list): list of nullable parameter names
                'enum' (list): list of parameters with enum values
                'validation' (list): list of parameters with validations
            root_map
                'validations' (dict): the dict mapping endpoint parameter tuple
                    paths to their validation dictionaries
                'allowed_values' (dict): the dict mapping endpoint parameter
                    tuple paths to their allowed_values (enum) dictionaries
                'openapi_types' (dict): param_name to openapi type
                'attribute_map' (dict): param_name to camelCase name
                'location_map' (dict): param_name to  'body', 'file', 'form',
                    'header', 'path', 'query'
                collection_format_map (dict): param_name to `csv` etc.
            headers_map (dict): see below key value pairs
                'accept' (list): list of Accept header strings
                'content_type' (list): list of Content-Type header strings
            api_client (ApiClient) api client instance
            callable (function): the function which is invoked when the
                Endpoint is called
        """
        self.settings = settings
        self.params_map = params_map
        self.params_map["all"].extend(
            [
                "async_req",
                "async_threadpool_executor",
                "_preload_content",
                "_request_timeout",
                "_return_http_data_only",
                "_check_input_type",
                "_check_return_type",
            ]
        )
        self.params_map["nullable"].extend(["_request_timeout"])
        self.validations = root_map["validations"]
        self.allowed_values = root_map["allowed_values"]
        self.openapi_types = root_map["openapi_types"]
        extra_types = {
            "async_req": (bool,),
            "async_threadpool_executor": (bool,),
            "_preload_content": (bool,),
            "_request_timeout": (none_type, float, (float,), [float], int, (int,), [int]),
            "_return_http_data_only": (bool,),
            "_check_input_type": (bool,),
            "_check_return_type": (bool,),
        }
        self.openapi_types.update(extra_types)
        self.attribute_map = root_map["attribute_map"]
        self.location_map = root_map["location_map"]
        self.collection_format_map = root_map["collection_format_map"]
        self.headers_map = headers_map
        self.api_client = api_client
        self.callable = callable

    def __call__(self, *args, **kwargs):
        """This method is invoked when endpoints are called
        Example:

        api_instance = InferenceApi()
        api_instance.embed  # this is an instance of the class Endpoint
        api_instance.embed()  # this invokes api_instance.embed.__call__()
        which then invokes the callable functions stored in that endpoint at
        api_instance.embed.callable or self.callable in this class

        """
        return self.callable(self, *args, **kwargs)

    def call_with_http_info(self, **kwargs):
        _host = self.api_client.configuration.host

        EndpointUtils.raise_if_unexpected_param(
            params_map=self.params_map, settings=self.settings, kwargs=kwargs
        )

        EndpointUtils.raise_if_missing_required_params(
            params_map=self.params_map, settings=self.settings, kwargs=kwargs
        )

        EndpointUtils.raise_if_invalid_inputs(
            config=self.api_client.configuration,
            params_map=self.params_map,
            allowed_values=self.allowed_values,
            validations=self.validations,
            openapi_types=self.openapi_types,
            kwargs=kwargs,
        )

        params = EndpointUtils.gather_params(
            attribute_map=self.attribute_map,
            location_map=self.location_map,
            openapi_types=self.openapi_types,
            collection_format_map=self.collection_format_map,
            kwargs=kwargs,
        )

        HeaderUtil.prepare_headers(headers_map=self.headers_map, params=params)

        return self.api_client.call_api(
            self.settings["endpoint_path"],
            self.settings["http_method"],
            path_params=params["path"],
            query_params=params["query"],
            header_params=params["header"],
            body=params["body"],
            post_params=params["form"],
            files=params["file"],
            response_type=self.settings["response_type"],
            auth_settings=self.settings["auth"],
            async_req=kwargs["async_req"],
            async_threadpool_executor=kwargs.get("async_threadpool_executor", None),
            _check_type=kwargs["_check_return_type"],
            _return_http_data_only=kwargs["_return_http_data_only"],
            _preload_content=kwargs["_preload_content"],
            _request_timeout=kwargs["_request_timeout"],
            _host=_host,
            collection_formats=params["collection_format"],
        )

    def _process_openapi_kwargs(
        self, kwargs: ExtraOpenApiKwargsTypedDict
    ) -> ExtraOpenApiKwargsTypedDict:
        kwargs["async_req"] = kwargs.get("async_req", False)
        kwargs["_return_http_data_only"] = kwargs.get("_return_http_data_only", True)
        kwargs["_preload_content"] = kwargs.get("_preload_content", True)
        kwargs["_request_timeout"] = kwargs.get("_request_timeout", None)
        kwargs["_check_input_type"] = kwargs.get("_check_input_type", True)
        kwargs["_check_return_type"] = kwargs.get("_check_return_type", True)
        return kwargs
