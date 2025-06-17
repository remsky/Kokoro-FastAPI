import logging
from typing import Optional, TYPE_CHECKING

from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
from pinecone.openapi_support.api_client import ApiClient

from pinecone.utils import setup_openapi_client, PluginAware
from pinecone.core.openapi.db_control import API_VERSION


logger = logging.getLogger(__name__)
""" @private """

if TYPE_CHECKING:
    from .resources.sync.index import IndexResource
    from .resources.sync.collection import CollectionResource
    from .resources.sync.restore_job import RestoreJobResource
    from .resources.sync.backup import BackupResource
    from pinecone.config import Config, OpenApiConfiguration


class DBControl(PluginAware):
    def __init__(
        self, config: "Config", openapi_config: "OpenApiConfiguration", pool_threads: int
    ) -> None:
        self.config = config
        """ @private """

        self._openapi_config = openapi_config
        """ @private """

        self._pool_threads = pool_threads
        """ @private """

        self._index_api = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=ManageIndexesApi,
            config=self.config,
            openapi_config=self._openapi_config,
            pool_threads=self._pool_threads,
            api_version=API_VERSION,
        )
        """ @private """

        self._index_resource: Optional["IndexResource"] = None
        """ @private """

        self._collection_resource: Optional["CollectionResource"] = None
        """ @private """

        self._restore_job_resource: Optional["RestoreJobResource"] = None
        """ @private """

        self._backup_resource: Optional["BackupResource"] = None
        """ @private """

        super().__init__()  # Initialize PluginAware

    @property
    def index(self) -> "IndexResource":
        if self._index_resource is None:
            from .resources.sync.index import IndexResource

            self._index_resource = IndexResource(
                index_api=self._index_api,
                config=self.config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._index_resource

    @property
    def collection(self) -> "CollectionResource":
        if self._collection_resource is None:
            from .resources.sync.collection import CollectionResource

            self._collection_resource = CollectionResource(
                index_api=self._index_api,
                config=self.config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._collection_resource

    @property
    def restore_job(self) -> "RestoreJobResource":
        if self._restore_job_resource is None:
            from .resources.sync.restore_job import RestoreJobResource

            self._restore_job_resource = RestoreJobResource(
                index_api=self._index_api,
                config=self.config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._restore_job_resource

    @property
    def backup(self) -> "BackupResource":
        if self._backup_resource is None:
            from .resources.sync.backup import BackupResource

            self._backup_resource = BackupResource(
                index_api=self._index_api,
                config=self.config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._backup_resource
