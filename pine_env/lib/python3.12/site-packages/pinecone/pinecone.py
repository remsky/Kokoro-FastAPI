import logging
from typing import Optional, Dict, Union, TYPE_CHECKING
from multiprocessing import cpu_count
import warnings

from pinecone.config import PineconeConfig, ConfigBuilder

from .legacy_pinecone_interface import LegacyPineconeDBControlInterface

from pinecone.utils import normalize_host, PluginAware, docslinks, require_kwargs
from .langchain_import_warnings import _build_langchain_attribute_error_message

logger = logging.getLogger(__name__)
""" @private """

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from pinecone.db_data import _Index as Index, _IndexAsyncio as IndexAsyncio
    from pinecone.db_control.index_host_store import IndexHostStore
    from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
    from pinecone.db_control.types import CreateIndexForModelEmbedTypedDict
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
    from pinecone.db_control.models import (
        ServerlessSpec,
        PodSpec,
        ByocSpec,
        IndexModel,
        IndexList,
        CollectionList,
        IndexEmbed,
        BackupModel,
        BackupList,
        RestoreJobModel,
        RestoreJobList,
    )


class Pinecone(PluginAware, LegacyPineconeDBControlInterface):
    """
    A client for interacting with Pinecone APIs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        proxy_url: Optional[str] = None,
        proxy_headers: Optional[Dict[str, str]] = None,
        ssl_ca_certs: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        pool_threads: Optional[int] = None,
        **kwargs,
    ):
        for deprecated_kwarg in {"config", "openapi_config", "index_api"}:
            if deprecated_kwarg in kwargs:
                raise NotImplementedError(
                    f"Passing {deprecated_kwarg} is no longer supported. Please pass individual settings such as proxy_url, proxy_headers, ssl_ca_certs, and ssl_verify directly to the Pinecone constructor as keyword arguments. See the README at {docslinks['README']} for examples."
                )

        self._config = PineconeConfig.build(
            api_key=api_key,
            host=host,
            additional_headers=additional_headers,
            proxy_url=proxy_url,
            proxy_headers=proxy_headers,
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
            **kwargs,
        )
        """ @private """

        self._openapi_config = ConfigBuilder.build_openapi_config(self._config, **kwargs)
        """ @private """

        if pool_threads is None:
            self._pool_threads = 5 * cpu_count()
            """ @private """
        else:
            self._pool_threads = pool_threads
            """ @private """

        self._inference = None  # Lazy initialization
        """ @private """

        self._db_control = None  # Lazy initialization
        """ @private """

        super().__init__()  # Initialize PluginAware

    @property
    def inference(self):
        """
        Inference is a namespace where an instance of the `pinecone.inference.Inference` class is lazily created and cached.
        """
        if self._inference is None:
            from pinecone.inference import Inference

            self._inference = Inference(
                config=self._config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._inference

    @property
    def db(self):
        """
        DBControl is a namespace where an instance of the `pinecone.db_control.DBControl` class is lazily created and cached.
        """
        if self._db_control is None:
            from pinecone.db_control import DBControl

            self._db_control = DBControl(
                config=self._config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._db_control

    @property
    def index_host_store(self) -> "IndexHostStore":
        """@private"""
        warnings.warn(
            "The `index_host_store` property is deprecated. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.db.index._index_host_store

    @property
    def config(self) -> "Config":
        """@private"""
        # The config property is considered private, but the name cannot be changed to include underscore
        # without breaking compatibility with plugins in the wild.
        return self._config

    @property
    def openapi_config(self) -> "OpenApiConfiguration":
        """@private"""
        warnings.warn(
            "The `openapi_config` property has been renamed to `_openapi_config`. It is considered private and should not be used directly. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._openapi_config

    @property
    def pool_threads(self) -> int:
        """@private"""
        warnings.warn(
            "The `pool_threads` property has been renamed to `_pool_threads`. It is considered private and should not be used directly. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._pool_threads

    @property
    def index_api(self) -> "ManageIndexesApi":
        """@private"""
        warnings.warn(
            "The `index_api` property is deprecated. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.db._index_api

    def create_index(
        self,
        name: str,
        spec: Union[Dict, "ServerlessSpec", "PodSpec", "ByocSpec"],
        dimension: Optional[int] = None,
        metric: Optional[Union["Metric", str]] = "cosine",
        timeout: Optional[int] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        vector_type: Optional[Union["VectorType", str]] = "dense",
        tags: Optional[Dict[str, str]] = None,
    ) -> "IndexModel":
        return self.db.index.create(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric,
            timeout=timeout,
            deletion_protection=deletion_protection,
            vector_type=vector_type,
            tags=tags,
        )

    def create_index_for_model(
        self,
        name: str,
        cloud: Union["CloudProvider", str],
        region: Union["AwsRegion", "GcpRegion", "AzureRegion", str],
        embed: Union["IndexEmbed", "CreateIndexForModelEmbedTypedDict"],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        timeout: Optional[int] = None,
    ) -> "IndexModel":
        return self.db.index.create_for_model(
            name=name,
            cloud=cloud,
            region=region,
            embed=embed,
            tags=tags,
            deletion_protection=deletion_protection,
            timeout=timeout,
        )

    @require_kwargs
    def create_index_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        tags: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> "IndexModel":
        return self.db.index.create_from_backup(
            name=name,
            backup_id=backup_id,
            deletion_protection=deletion_protection,
            tags=tags,
            timeout=timeout,
        )

    def delete_index(self, name: str, timeout: Optional[int] = None):
        return self.db.index.delete(name=name, timeout=timeout)

    def list_indexes(self) -> "IndexList":
        return self.db.index.list()

    def describe_index(self, name: str) -> "IndexModel":
        return self.db.index.describe(name=name)

    def has_index(self, name: str) -> bool:
        return self.db.index.has(name=name)

    def configure_index(
        self,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[Union["PodType", str]] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        return self.db.index.configure(
            name=name,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
        )

    def create_collection(self, name: str, source: str) -> None:
        return self.db.collection.create(name=name, source=source)

    def list_collections(self) -> "CollectionList":
        return self.db.collection.list()

    def delete_collection(self, name: str) -> None:
        return self.db.collection.delete(name=name)

    def describe_collection(self, name: str):
        return self.db.collection.describe(name=name)

    @require_kwargs
    def create_backup(
        self, *, index_name: str, backup_name: str, description: str = ""
    ) -> "BackupModel":
        return self.db.backup.create(
            index_name=index_name, backup_name=backup_name, description=description
        )

    @require_kwargs
    def list_backups(
        self,
        *,
        index_name: Optional[str] = None,
        limit: Optional[int] = 10,
        pagination_token: Optional[str] = None,
    ) -> "BackupList":
        return self.db.backup.list(
            index_name=index_name, limit=limit, pagination_token=pagination_token
        )

    @require_kwargs
    def describe_backup(self, *, backup_id: str) -> "BackupModel":
        return self.db.backup.describe(backup_id=backup_id)

    @require_kwargs
    def delete_backup(self, *, backup_id: str) -> None:
        return self.db.backup.delete(backup_id=backup_id)

    @require_kwargs
    def list_restore_jobs(
        self, *, limit: Optional[int] = 10, pagination_token: Optional[str] = None
    ) -> "RestoreJobList":
        return self.db.restore_job.list(limit=limit, pagination_token=pagination_token)

    @require_kwargs
    def describe_restore_job(self, *, job_id: str) -> "RestoreJobModel":
        return self.db.restore_job.describe(job_id=job_id)

    @staticmethod
    def from_texts(*args, **kwargs):
        """@private"""
        raise AttributeError(_build_langchain_attribute_error_message("from_texts"))

    @staticmethod
    def from_documents(*args, **kwargs):
        """@private"""
        raise AttributeError(_build_langchain_attribute_error_message("from_documents"))

    def Index(self, name: str = "", host: str = "", **kwargs) -> "Index":
        from pinecone.db_data import _Index

        if name == "" and host == "":
            raise ValueError("Either name or host must be specified")

        pt = kwargs.pop("pool_threads", None) or self._pool_threads
        api_key = self._config.api_key
        openapi_config = self._openapi_config

        if host != "":
            check_realistic_host(host)

            # Use host url if it is provided
            index_host = normalize_host(host)
        else:
            # Otherwise, get host url from describe_index using the index name
            index_host = self.db.index._get_host(name)

        return _Index(
            host=index_host,
            api_key=api_key,
            pool_threads=pt,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs,
        )

    def IndexAsyncio(self, host: str, **kwargs) -> "IndexAsyncio":
        from pinecone.db_data import _IndexAsyncio

        api_key = self._config.api_key
        openapi_config = self._openapi_config

        if host is None or host == "":
            raise ValueError("A host must be specified")

        check_realistic_host(host)
        index_host = normalize_host(host)

        return _IndexAsyncio(
            host=index_host,
            api_key=api_key,
            openapi_config=openapi_config,
            source_tag=self.config.source_tag,
            **kwargs,
        )


def check_realistic_host(host: str) -> None:
    """@private

    Checks whether a user-provided host string seems plausible.
    Someone could erroneously pass an index name as the host by
    mistake, and if they have done that we'd like to give them a
    simple error message as feedback rather than attempting to
    call the url and getting a more cryptic DNS resolution error.
    """

    if "." not in host and "localhost" not in host:
        raise ValueError(
            f"You passed '{host}' as the host but this does not appear to be valid. Call describe_index() to confirm the host of the index."
        )
