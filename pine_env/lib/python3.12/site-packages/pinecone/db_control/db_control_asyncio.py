import logging
from typing import Optional, TYPE_CHECKING

from pinecone.core.openapi.db_control.api.manage_indexes_api import AsyncioManageIndexesApi
from pinecone.openapi_support import AsyncioApiClient

from pinecone.utils import setup_async_openapi_client
from pinecone.core.openapi.db_control import API_VERSION

logger = logging.getLogger(__name__)
""" @private """


if TYPE_CHECKING:
    from .resources.asyncio.index import IndexResourceAsyncio
    from .resources.asyncio.collection import CollectionResourceAsyncio
    from .resources.asyncio.restore_job import RestoreJobResourceAsyncio
    from .resources.asyncio.backup import BackupResourceAsyncio
    from pinecone.config import Config, OpenApiConfiguration


class DBControlAsyncio:
    def __init__(self, config: "Config", openapi_config: "OpenApiConfiguration") -> None:
        self._config = config
        """ @private """

        self._openapi_config = openapi_config
        """ @private """

        self._index_api = setup_async_openapi_client(
            api_client_klass=AsyncioApiClient,
            api_klass=AsyncioManageIndexesApi,
            config=self._config,
            openapi_config=self._openapi_config,
            api_version=API_VERSION,
        )
        """ @private """

        self._index_resource: Optional["IndexResourceAsyncio"] = None
        """ @private """

        self._collection_resource: Optional["CollectionResourceAsyncio"] = None
        """ @private """

        self._restore_job_resource: Optional["RestoreJobResourceAsyncio"] = None
        """ @private """

        self._backup_resource: Optional["BackupResourceAsyncio"] = None
        """ @private """

    @property
    def index(self) -> "IndexResourceAsyncio":
        if self._index_resource is None:
            from .resources.asyncio.index import IndexResourceAsyncio

            self._index_resource = IndexResourceAsyncio(
                index_api=self._index_api, config=self._config
            )
        return self._index_resource

    @property
    def collection(self) -> "CollectionResourceAsyncio":
        if self._collection_resource is None:
            from .resources.asyncio.collection import CollectionResourceAsyncio

            self._collection_resource = CollectionResourceAsyncio(self._index_api)
        return self._collection_resource

    @property
    def restore_job(self) -> "RestoreJobResourceAsyncio":
        if self._restore_job_resource is None:
            from .resources.asyncio.restore_job import RestoreJobResourceAsyncio

            self._restore_job_resource = RestoreJobResourceAsyncio(self._index_api)
        return self._restore_job_resource

    @property
    def backup(self) -> "BackupResourceAsyncio":
        if self._backup_resource is None:
            from .resources.asyncio.backup import BackupResourceAsyncio

            self._backup_resource = BackupResourceAsyncio(self._index_api)
        return self._backup_resource
