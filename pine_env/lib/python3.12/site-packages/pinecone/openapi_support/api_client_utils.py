import json
import mimetypes
import io
import os
from urllib3.fields import RequestField
from urllib.parse import quote

from typing import Optional, List, Tuple, Dict, Any, Union
from .serializer import Serializer
from .exceptions import PineconeApiValueError


class HeaderUtil:
    @staticmethod
    def select_header_content_type(content_types: List[str]) -> str:
        """Returns `Content-Type` based on an array of content_types provided.

        :param content_types: List of content-types.
        :return: Content-Type (e.g. application/json).
        """
        if not content_types:
            return "application/json"

        content_types = [x.lower() for x in content_types]

        if "application/json" in content_types or "*/*" in content_types:
            return "application/json"
        else:
            return content_types[0]

    @staticmethod
    def select_header_accept(accepts: List[str]) -> str:
        """Returns `Accept` based on an array of accepts provided.

        :param accepts: List of headers.
        :return: Accept (e.g. application/json).
        """
        if not accepts:
            return ""

        accepts = [x.lower() for x in accepts]

        if "application/json" in accepts:
            return "application/json"
        else:
            return ", ".join(accepts)

    @staticmethod
    def process_header_params(
        default_headers: Dict[str, str], header_params: Dict[str, str], collection_formats
    ) -> Dict[str, Any]:
        header_params.update(default_headers)
        if header_params:
            sanitized_header_params: Dict[str, Any] = Serializer.sanitize_for_serialization(
                header_params
            )
            processed_header_params: Dict[str, Any] = dict(
                parameters_to_tuples(sanitized_header_params, collection_formats)
            )
        return processed_header_params

    @staticmethod
    def prepare_headers(headers_map: Dict[str, List[str]], params) -> None:
        """Mutates the params to set Accept and Content-Type headers."""
        accept_headers_list = headers_map["accept"]
        if accept_headers_list:
            params["header"]["Accept"] = HeaderUtil.select_header_accept(accept_headers_list)

        content_type_headers_list = headers_map["content_type"]
        if content_type_headers_list:
            header_list = HeaderUtil.select_header_content_type(content_type_headers_list)
            params["header"]["Content-Type"] = header_list


def process_query_params(query_params, collection_formats):
    if query_params:
        sanitized_query_params = Serializer.sanitize_for_serialization(query_params)
        processed_query_params = parameters_to_tuples(sanitized_query_params, collection_formats)
    else:
        processed_query_params = []

    return processed_query_params


def process_params(
    default_headers: Dict[str, str],
    header_params: Dict[str, Any],
    path_params: Dict[str, Any],
    collection_formats: Dict[str, str],
):
    # header parameters
    headers_tuple = HeaderUtil.process_header_params(
        default_headers, header_params, collection_formats
    )

    # path parameters
    sanitized_path_params: Dict[str, Any] = Serializer.sanitize_for_serialization(path_params or {})
    path_parm: List[Tuple[str, Any]] = parameters_to_tuples(
        sanitized_path_params, collection_formats
    )

    return headers_tuple, path_parm, sanitized_path_params


def parameters_to_multipart(params, collection_types):
    """Get parameters as list of tuples, formatting as json if value is collection_types

    :param params: Parameters as list of two-tuples
    :param dict collection_types: Parameter collection types
    :return: Parameters as list of tuple or urllib3.fields.RequestField
    """
    new_params = []
    if collection_types is None:
        collection_types = dict
    for k, v in params.items() if isinstance(params, dict) else params:  # noqa: E501
        if isinstance(
            v, collection_types
        ):  # v is instance of collection_type, formatting as application/json
            v = json.dumps(v, ensure_ascii=False).encode("utf-8")
            field = RequestField(k, v)
            field.make_multipart(content_type="application/json; charset=utf-8")
            new_params.append(field)
        else:
            new_params.append((k, v))
    return new_params


def files_parameters(files: Optional[Dict[str, List[io.IOBase]]] = None):
    """Builds form parameters.

    :param files: None or a dict with key=param_name and
        value is a list of open file objects
    :return: List of tuples of form parameters with file data
    """
    if files is None:
        return []

    params = []
    for param_name, file_instances in files.items():
        if file_instances is None:
            # if the file field is nullable, skip None values
            continue
        for file_instance in file_instances:
            if file_instance is None:
                # if the file field is nullable, skip None values
                continue
            if file_instance.closed is True:
                raise PineconeApiValueError(
                    "Cannot read a closed file. The passed in file_type "
                    "for %s must be open." % param_name
                )
            filename = os.path.basename(file_instance.name)  # type: ignore
            filedata = Serializer.get_file_data_and_close_file(file_instance)
            mimetype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            params.append(tuple([param_name, tuple([filename, filedata, mimetype])]))

    return params


def parameters_to_tuples(
    params: Union[Dict[str, Any], List[Tuple[str, Any]]],
    collection_formats: Optional[Dict[str, str]],
) -> List[Tuple[str, str]]:
    """Get parameters as list of tuples, formatting collections.

    :param params: Parameters as dict or list of two-tuples
    :param dict collection_formats: Parameter collection formats
    :return: Parameters as list of tuples, collections formatted
    """
    new_params: List[Tuple[str, Any]] = []
    if collection_formats is None:
        collection_formats = {}
    for k, v in params.items() if isinstance(params, dict) else params:  # noqa: E501
        if k in collection_formats:
            collection_format = collection_formats[k]
            if collection_format == "multi":
                new_params.extend((k, value) for value in v)
            else:
                if collection_format == "ssv":
                    delimiter = " "
                elif collection_format == "tsv":
                    delimiter = "\t"
                elif collection_format == "pipes":
                    delimiter = "|"
                else:  # csv is the default
                    delimiter = ","
                new_params.append((k, delimiter.join(str(value) for value in v)))
        else:
            new_params.append((k, v))
    return new_params


def build_request_url(config, processed_path_params, resource_path, _host):
    for k, v in processed_path_params:
        # specified safe chars, encode everything
        resource_path = resource_path.replace(
            "{%s}" % k, quote(str(v), safe=config.safe_chars_for_path_param)
        )

    # _host is a host override passed for an individual operation
    host = _host if _host is not None else config.host

    return host + resource_path
