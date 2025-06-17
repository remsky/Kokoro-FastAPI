from enum import Enum
from typing import Optional, TypedDict, Any, Union

from pinecone.core.openapi.db_data.models import (
    StartImportRequest,
    ImportErrorMode as ImportErrorModeClass,
)

from pinecone.utils import parse_non_empty_args, convert_enum_to_string


class ImportErrorMode(Enum):
    CONTINUE = "CONTINUE"
    ABORT = "ABORT"


class DescribeImportArgs(TypedDict, total=False):
    id: str


class CancelImportArgs(TypedDict, total=False):
    id: str


class BulkImportRequestFactory:
    @staticmethod
    def start_import_request(
        uri: str,
        integration_id: Optional[str] = None,
        error_mode: Optional[Union[ImportErrorMode, str]] = "CONTINUE",
    ) -> StartImportRequest:
        if error_mode is None:
            error_mode = "CONTINUE"
        error_mode_str = convert_enum_to_string(error_mode).lower()
        valid_error_modes = [mode.value.lower() for mode in ImportErrorMode]
        if error_mode_str not in valid_error_modes:
            raise ValueError(
                f"Invalid error_mode: {error_mode}. Must be one of {valid_error_modes}"
            )

        args_dict = parse_non_empty_args(
            [
                ("uri", uri),
                ("integration_id", integration_id),
                ("error_mode", ImportErrorModeClass(on_error=error_mode_str)),
            ]
        )

        return StartImportRequest(**args_dict)

    @staticmethod
    def list_imports_paginated_args(
        limit: Optional[int] = None, pagination_token: Optional[str] = None, **kwargs
    ) -> dict[str, Any]:
        return parse_non_empty_args([("limit", limit), ("pagination_token", pagination_token)])

    @staticmethod
    def describe_import_args(id: str) -> DescribeImportArgs:
        if isinstance(id, int):
            id = str(id)
        return {"id": id}

    @staticmethod
    def cancel_import_args(id: str) -> CancelImportArgs:
        if isinstance(id, int):
            id = str(id)
        return {"id": id}
