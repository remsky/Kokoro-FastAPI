import logging
import warnings
from typing import Optional, Dict, Union, TYPE_CHECKING

from pinecone.config import PineconeConfig, ConfigBuilder

from pinecone.utils import normalize_host, require_kwargs, docslinks

from .pinecone_interface_asyncio import PineconeAsyncioDBControlInterface
from .pinecone import check_realistic_host

if TYPE_CHECKING:
    from pinecone.db_control.types import CreateIndexForModelEmbedTypedDict
    from pinecone.db_data import _IndexAsyncio
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
    from pinecone.core.openapi.db_control.api.manage_indexes_api import AsyncioManageIndexesApi
    from pinecone.db_control.index_host_store import IndexHostStore

logger = logging.getLogger(__name__)
""" @private """


class PineconeAsyncio(PineconeAsyncioDBControlInterface):
    """
    `PineconeAsyncio` is an asyncio client for interacting with Pinecone's control plane API.

    This class implements methods for managing and interacting with Pinecone resources
    such as collections and indexes.

    To perform data operations such as inserting and querying vectors, use the `IndexAsyncio` class.

    ```python
    import asyncio
    from pinecone import Pinecone

    async def main():
        pc = Pinecone()
        async with pc.IndexAsyncio(host="my-index.pinecone.io") as idx:
            await idx.upsert(vectors=[(1, [1, 2, 3]), (2, [4, 5, 6])])

    asyncio.run(main())
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        proxy_url: Optional[str] = None,
        # proxy_headers: Optional[Dict[str, str]] = None,
        ssl_ca_certs: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        **kwargs,
    ):
        for deprecated_kwarg in {"config", "openapi_config"}:
            if deprecated_kwarg in kwargs:
                raise NotImplementedError(
                    f"Passing {deprecated_kwarg} is no longer supported. Please pass individual settings such as proxy_url, ssl_ca_certs, and ssl_verify directly to the Pinecone constructor as keyword arguments. See the README at {docslinks['README']} for examples."
                )

        for unimplemented_kwarg in {"proxy_headers"}:
            if unimplemented_kwarg in kwargs:
                raise NotImplementedError(
                    f"You have passed {unimplemented_kwarg} but this configuration has not been implemented for PineconeAsyncio."
                )

        self._config = PineconeConfig.build(
            api_key=api_key,
            host=host,
            additional_headers=additional_headers,
            proxy_url=proxy_url,
            proxy_headers=None,
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
            **kwargs,
        )
        """ @private """

        self._openapi_config = ConfigBuilder.build_openapi_config(self._config, **kwargs)
        """ @private """

        self._inference = None  # Lazy initialization
        """ @private """

        self._db_control = None  # Lazy initialization
        """ @private """

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def close(self):
        """Cleanup resources used by the Pinecone client.

        This method should be called when the client is no longer needed so that
        it can cleanup the aioahttp session and other resources.

        After close has been called, the client instance should not be used.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            pc = PineconeAsyncio()
            desc = await pc.describe_index(name="my-index")
            await pc.close()

        asyncio.run(main())
        ```

        If you are using the client as a context manager, the close method is called automatically
        when exiting.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio() as pc:
                desc = await pc.describe_index(name="my-index")

        # No need to call close in this case because the "async with" syntax
        # automatically calls close when exiting the block.
        asyncio.run(main())
        ```

        """
        await self.db._index_api.api_client.close()

    @property
    def inference(self):
        """Dynamically create and cache the AsyncioInference instance."""
        if self._inference is None:
            from pinecone.inference import AsyncioInference

            self._inference = AsyncioInference(api_client=self.db._index_api.api_client)
        return self._inference

    @property
    def db(self):
        if self._db_control is None:
            from .db_control.db_control_asyncio import DBControlAsyncio

            self._db_control = DBControlAsyncio(
                config=self._config, openapi_config=self._openapi_config
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
    def index_api(self) -> "AsyncioManageIndexesApi":
        """@private"""
        warnings.warn(
            "The `index_api` property is deprecated. This warning will become an error in a future version of the Pinecone Python SDK.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.db._index_api

    async def create_index(
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
        resp = await self.db.index.create(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric,
            deletion_protection=deletion_protection,
            vector_type=vector_type,
            tags=tags,
            timeout=timeout,
        )
        return resp

    async def create_index_for_model(
        self,
        name: str,
        cloud: Union["CloudProvider", str],
        region: Union["AwsRegion", "GcpRegion", "AzureRegion", str],
        embed: Union["IndexEmbed", "CreateIndexForModelEmbedTypedDict"],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        timeout: Optional[int] = None,
    ) -> "IndexModel":
        return await self.db.index.create_for_model(
            name=name,
            cloud=cloud,
            region=region,
            embed=embed,
            tags=tags,
            deletion_protection=deletion_protection,
            timeout=timeout,
        )

    @require_kwargs
    async def create_index_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        tags: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> "IndexModel":
        return await self.db.index.create_from_backup(
            name=name,
            backup_id=backup_id,
            deletion_protection=deletion_protection,
            tags=tags,
            timeout=timeout,
        )

    async def delete_index(self, name: str, timeout: Optional[int] = None):
        return await self.db.index.delete(name=name, timeout=timeout)

    async def list_indexes(self) -> "IndexList":
        return await self.db.index.list()

    async def describe_index(self, name: str) -> "IndexModel":
        return await self.db.index.describe(name=name)

    async def has_index(self, name: str) -> bool:
        return await self.db.index.has(name=name)

    async def configure_index(
        self,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[Union["PodType", str]] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        return await self.db.index.configure(
            name=name,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
        )

    async def create_collection(self, name: str, source: str):
        return await self.db.collection.create(name=name, source=source)

    async def list_collections(self) -> "CollectionList":
        return await self.db.collection.list()

    async def delete_collection(self, name: str):
        return await self.db.collection.delete(name=name)

    async def describe_collection(self, name: str):
        return await self.db.collection.describe(name=name)

    @require_kwargs
    async def create_backup(
        self, *, index_name: str, backup_name: str, description: str = ""
    ) -> "BackupModel":
        return await self.db.backup.create(
            index_name=index_name, backup_name=backup_name, description=description
        )

    @require_kwargs
    async def list_backups(
        self,
        *,
        index_name: Optional[str] = None,
        limit: Optional[int] = 10,
        pagination_token: Optional[str] = None,
    ) -> "BackupList":
        return await self.db.backup.list(
            index_name=index_name, limit=limit, pagination_token=pagination_token
        )

    @require_kwargs
    async def describe_backup(self, *, backup_id: str) -> "BackupModel":
        return await self.db.backup.describe(backup_id=backup_id)

    @require_kwargs
    async def delete_backup(self, *, backup_id: str) -> None:
        return await self.db.backup.delete(backup_id=backup_id)

    @require_kwargs
    async def list_restore_jobs(
        self, *, limit: Optional[int] = 10, pagination_token: Optional[str] = None
    ) -> "RestoreJobList":
        return await self.db.restore_job.list(limit=limit, pagination_token=pagination_token)

    @require_kwargs
    async def describe_restore_job(self, *, job_id: str) -> "RestoreJobModel":
        return await self.db.restore_job.describe(job_id=job_id)

    def IndexAsyncio(self, host: str, **kwargs) -> "_IndexAsyncio":
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
            source_tag=self._config.source_tag,
            **kwargs,
        )
