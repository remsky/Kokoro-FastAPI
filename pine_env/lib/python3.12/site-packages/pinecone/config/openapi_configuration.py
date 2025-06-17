import copy
import logging
import multiprocessing

from pinecone.exceptions import PineconeApiValueError
from typing import TypedDict


class HostSetting(TypedDict):
    url: str
    description: str


JSON_SCHEMA_VALIDATION_KEYWORDS = {
    "multipleOf",
    "maximum",
    "exclusiveMaximum",
    "minimum",
    "exclusiveMinimum",
    "maxLength",
    "minLength",
    "pattern",
    "maxItems",
    "minItems",
}


class Configuration:
    """Class to hold the configuration of the API client.

        :param host: Base url
        :param api_key: Dict to store API key(s).
          Each entry in the dict specifies an API key.
          The dict key is the name of the security scheme in the OAS specification.
          The dict value is the API key secret.
        :param api_key_prefix: Dict to store API prefix (e.g. Bearer)
          The dict key is the name of the security scheme in the OAS specification.
          The dict value is an API key prefix when generating the auth data.
        :param discard_unknown_keys: Boolean value indicating whether to discard
          unknown properties. A server may send a response that includes additional
          properties that are not known by the client in the following scenarios:
          1. The OpenAPI document is incomplete, i.e. it does not match the server
             implementation.
          2. The client was generated using an older version of the OpenAPI document
             and the server has been upgraded since then.
          If a schema in the OpenAPI document defines the additionalProperties attribute,
          then all undeclared properties received by the server are injected into the
          additional properties map. In that case, there are undeclared properties, and
          nothing to discard.
        :param disabled_client_side_validations (string): Comma-separated list of
          JSON schema validation keywords to disable JSON schema structural validation
          rules. The following keywords may be specified: multipleOf, maximum,
          exclusiveMaximum, minimum, exclusiveMinimum, maxLength, minLength, pattern,
          maxItems, minItems.
          By default, the validation is performed for data generated locally by the client
          and data received from the server, independent of any validation performed by
          the server side. If the input data does not satisfy the JSON schema validation
          rules specified in the OpenAPI document, an exception is raised.
          If disabled_client_side_validations is set, structural validation is
          disabled. This can be useful to troubleshoot data validation problem, such as
          when the OpenAPI document validation rules do not match the actual API data
          received by the server.
        :param server_operation_index: Mapping from operation ID to an index to server
          configuration.
        :param server_operation_variables: Mapping from operation ID to a mapping with
          string values to replace variables in templated server configuration.
          The validation of enums is performed for variables with defined enum values before.
        :param ssl_ca_cert: str - the path to a file of concatenated CA certificates
          in PEM format

        :Example:

        API Key Authentication Example.
        Given the following security scheme in the OpenAPI specification:
          components:
            securitySchemes:
              cookieAuth:         # name for the security scheme
                type: apiKey
                in: cookie
                name: JSESSIONID  # cookie name

        You can programmatically set the cookie:

    conf = pinecone.config.openapi_configuration.Configuration(
        api_key={'cookieAuth': 'abc123'}
        api_key_prefix={'cookieAuth': 'JSESSIONID'}
    )

        The following cookie will be added to the HTTP request:
           Cookie: JSESSIONID abc123
    """

    _default = None

    def __init__(
        self,
        host=None,
        api_key=None,
        api_key_prefix=None,
        discard_unknown_keys=False,
        disabled_client_side_validations="",
        server_index=None,
        server_variables=None,
        server_operation_index=None,
        server_operation_variables=None,
        ssl_ca_cert=None,
    ):
        """Constructor"""
        self._base_path = "https://api.pinecone.io" if host is None else host
        """Default Base url
        """
        self.server_index = 0 if server_index is None and host is None else server_index
        self.server_operation_index = server_operation_index or {}
        """Default server index
        """
        self.server_variables = server_variables or {}
        self.server_operation_variables = server_operation_variables or {}
        """Default server variables
        """
        self.temp_folder_path = None
        """Temp file folder for downloading files
        """
        # Authentication Settings
        self.api_key = {}
        if api_key:
            self.api_key = api_key
        """dict to store API key(s)
        """
        self.api_key_prefix = {}
        if api_key_prefix:
            self.api_key_prefix = api_key_prefix
        """dict to store API prefix (e.g. Bearer)
        """
        self.refresh_api_key_hook = None
        """function hook to refresh API key if expired
        """
        self.discard_unknown_keys = discard_unknown_keys
        self.disabled_client_side_validations = disabled_client_side_validations
        self.logger = {}
        """Logging Settings
        """
        self.logger["package_logger"] = logging.getLogger("pinecone.openapi_support")
        self.logger["urllib3_logger"] = logging.getLogger("urllib3")
        self.logger_format = "%(asctime)s %(levelname)s %(message)s"
        """Log format
        """
        self.logger_stream_handler = None
        """Log stream handler
        """
        self.logger_file_handler = None
        """Log file handler
        """
        self.logger_file = None
        """Debug file location
        """
        # Initialize debug directly without using the property setter
        self.debug = False
        """Debug switch
        """

        self.verify_ssl = True
        """SSL/TLS verification
           Set this to false to skip verifying SSL certificate when calling API
           from https server.
        """
        self.ssl_ca_cert = ssl_ca_cert
        """Set this to customize the certificate file to verify the peer.
        """
        self.cert_file = None
        """client certificate file
        """
        self.key_file = None
        """client key file
        """
        self.assert_hostname = None
        """Set this to True/False to enable/disable SSL hostname verification.
        """

        self.connection_pool_maxsize = multiprocessing.cpu_count() * 5
        """urllib3 connection pool's maximum number of connections saved
           per pool. urllib3 uses 1 connection as default value, but this is
           not the best value when you are making a lot of possibly parallel
           requests to the same host, which is often the case here.
           cpu_count * 5 is used as default value to increase performance.
        """

        self.proxy = None
        """Proxy URL
        """
        self.proxy_headers = None
        """Proxy headers
        """
        self.safe_chars_for_path_param = ""
        """Safe chars for path_param
        """
        self.retries = None
        """Adding retries to override urllib3 default value 3
        """
        # Enable client side validation
        self.client_side_validation = True

        # Options to pass down to the underlying urllib3 socket
        self.socket_options = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ("logger", "logger_file_handler"):
                setattr(result, k, copy.deepcopy(v, memo))
        # shallow copy of loggers
        result.logger = copy.copy(self.logger)
        # use setters to configure loggers
        result.logger_file = self.logger_file
        result.debug = self.debug
        return result

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "disabled_client_side_validations":
            s = set(filter(None, value.split(",")))
            for v in s:
                if v not in JSON_SCHEMA_VALIDATION_KEYWORDS:
                    raise PineconeApiValueError("Invalid keyword: '{0}''".format(v))
            self._disabled_client_side_validations = s

    @classmethod
    def set_default(cls, default):
        """Set default instance of configuration.

        It stores default configuration, which can be
        returned by get_default_copy method.

        :param default: object of Configuration
        """
        cls._default = copy.deepcopy(default)

    @classmethod
    def get_default_copy(cls):
        """Return new instance of configuration.

        This method returns newly created, based on default constructor,
        object of Configuration class or returns a copy of default
        configuration passed by the set_default method.

        :return: The configuration object.
        """
        if cls._default is not None:
            return copy.deepcopy(cls._default)
        return Configuration()

    @property
    def logger_file(self):
        """The logger file.

        If the logger_file is None, then add stream handler and remove file
        handler. Otherwise, add file handler and remove stream handler.

        :param value: The logger_file path.
        :type: str
        """
        return self.__logger_file

    @logger_file.setter
    def logger_file(self, value):
        """The logger file.

        If the logger_file is None, then add stream handler and remove file
        handler. Otherwise, add file handler and remove stream handler.

        :param value: The logger_file path.
        :type: str
        """
        self.__logger_file = value
        if self.__logger_file:
            # If set logging file,
            # then add file handler and remove stream handler.
            self.logger_file_handler = logging.FileHandler(self.__logger_file)
            self.logger_file_handler.setFormatter(self.logger_formatter)
            for _, logger in self.logger.items():
                logger.addHandler(self.logger_file_handler)

    @property
    def debug(self):
        """Debug status

        :param value: The debug status, True or False.
        :type: bool
        """
        return self._debug

    @debug.setter
    def debug(self, value):
        """Debug status

        :param value: The debug status, True or False.
        :type: bool
        """
        if hasattr(self, "_debug"):
            previous_debug = self._debug
        else:
            previous_debug = None
        self._debug = value

        def enable_http_logging():
            from http import client as http_client

            http_client.HTTPConnection.debuglevel = 1

        def disable_http_logging():
            from http import client as http_client

            http_client.HTTPConnection.debuglevel = 0

        def set_default_log_level(c):
            for _, logger in c.logger.items():
                logger.setLevel(logging.WARNING)

        if self._debug:
            for _, logger in self.logger.items():
                logger.setLevel(logging.DEBUG)
            enable_http_logging()
        elif previous_debug is True and self._debug is False:
            set_default_log_level(self)
            disable_http_logging()
        else:
            # On the initial call, we don't need to do anything to http
            # logging, since it's not enabled by default.
            set_default_log_level(self)

    @property
    def logger_format(self):
        """The logger format.

        The logger_formatter will be updated when sets logger_format.

        :param value: The format string.
        :type: str
        """
        return self.__logger_format

    @logger_format.setter
    def logger_format(self, value):
        """The logger format.

        The logger_formatter will be updated when sets logger_format.

        :param value: The format string.
        :type: str
        """
        self.__logger_format = value
        self.logger_formatter = logging.Formatter(self.__logger_format)

    def get_api_key_with_prefix(self, identifier, alias=None):
        """Gets API key (with prefix if set).

        :param identifier: The identifier of apiKey.
        :param alias: The alternative identifier of apiKey.
        :return: The token for api key authentication.
        """
        if self.refresh_api_key_hook is not None:
            self.refresh_api_key_hook(self)
        key = self.api_key.get(identifier, self.api_key.get(alias) if alias is not None else None)
        if key:
            prefix = self.api_key_prefix.get(identifier)
            if prefix:
                return "%s %s" % (prefix, key)
            else:
                return key

    def auth_settings(self):
        """Gets Auth Settings dict for api client.

        :return: The Auth Settings information dict.
        """
        auth = {}
        if "ApiKeyAuth" in self.api_key:
            auth["ApiKeyAuth"] = {
                "type": "api_key",
                "in": "header",
                "key": "Api-Key",
                "value": self.get_api_key_with_prefix("ApiKeyAuth"),
            }
        return auth

    def get_host_settings(self):
        """Gets an array of host settings

        :return: An array of host settings
        """
        return [{"url": "https://api.pinecone.io", "description": "Production API endpoints"}]

    def get_host_from_settings(self, index, variables=None, servers=None):
        """Gets host URL based on the index and variables
        :param index: array index of the host settings
        :param variables: hash of variable and the corresponding value
        :param servers: an array of host settings or None
        :return: URL based on host settings
        """
        if index is None:
            return self._base_path

        variables = {} if variables is None else variables
        servers = self.get_host_settings() if servers is None else servers

        try:
            server = servers[index]
        except IndexError:
            raise ValueError(
                "Invalid index {0} when selecting the host settings. Must be less than {1}".format(
                    index, len(servers)
                )
            )

        url = server["url"]

        # go through variables and replace placeholders
        for variable_name, variable in server.get("variables", {}).items():
            used_value = variables.get(variable_name, variable["default_value"])

            if "enum_values" in variable and used_value not in variable["enum_values"]:
                raise ValueError(
                    "The variable `{0}` in the host URL has invalid value {1}. Must be {2}.".format(
                        variable_name, variables[variable_name], variable["enum_values"]
                    )
                )

            url = url.replace("{" + variable_name + "}", used_value)

        return url

    @property
    def host(self):
        """Return generated host."""
        return self.get_host_from_settings(self.server_index, variables=self.server_variables)

    @host.setter
    def host(self, value):
        """Fix base path."""
        self._base_path = value
        self.server_index = None

    def __repr__(self):
        attrs = [
            f"host={self.host}",
            "api_key=***",
            f"api_key_prefix={self.api_key_prefix}",
            f"connection_pool_maxsize={self.connection_pool_maxsize}",
            f"discard_unknown_keys={self.discard_unknown_keys}",
            f"disabled_client_side_validations={self.disabled_client_side_validations}",
            f"server_index={self.server_index}",
            f"server_variables={self.server_variables}",
            f"server_operation_index={self.server_operation_index}",
            f"server_operation_variables={self.server_operation_variables}",
            f"ssl_ca_cert={self.ssl_ca_cert}",
        ]
        return f"Configuration({', '.join(attrs)})"
