import logging
import asyncio
from typing import Optional, Dict, Union


from pinecone.db_control.models import (
    ServerlessSpec,
    PodSpec,
    ByocSpec,
    IndexModel,
    IndexList,
    IndexEmbed,
)
from pinecone.utils import docslinks

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
from pinecone.db_control.types import CreateIndexForModelEmbedTypedDict
from pinecone.db_control.request_factory import PineconeDBControlRequestFactory
from pinecone.core.openapi.db_control import API_VERSION
from pinecone.utils import require_kwargs

logger = logging.getLogger(__name__)
""" @private """


class IndexResourceAsyncio:
    def __init__(self, index_api, config):
        self._index_api = index_api
        self._config = config

    @require_kwargs
    async def create(
        self,
        *,
        name: str,
        spec: Union[Dict, ServerlessSpec, PodSpec, ByocSpec],
        dimension: Optional[int] = None,
        metric: Optional[Union[Metric, str]] = Metric.COSINE,
        timeout: Optional[int] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        vector_type: Optional[Union[VectorType, str]] = VectorType.DENSE,
        tags: Optional[Dict[str, str]] = None,
    ) -> IndexModel:
        req = PineconeDBControlRequestFactory.create_index_request(
            name=name,
            spec=spec,
            dimension=dimension,
            metric=metric,
            deletion_protection=deletion_protection,
            vector_type=vector_type,
            tags=tags,
        )
        resp = await self._index_api.create_index(create_index_request=req)

        if timeout == -1:
            return IndexModel(resp)
        return await self.__poll_describe_index_until_ready(name, timeout)

    @require_kwargs
    async def create_for_model(
        self,
        *,
        name: str,
        cloud: Union[CloudProvider, str],
        region: Union[AwsRegion, GcpRegion, AzureRegion, str],
        embed: Union[IndexEmbed, CreateIndexForModelEmbedTypedDict],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        timeout: Optional[int] = None,
    ) -> IndexModel:
        req = PineconeDBControlRequestFactory.create_index_for_model_request(
            name=name,
            cloud=cloud,
            region=region,
            embed=embed,
            tags=tags,
            deletion_protection=deletion_protection,
        )
        resp = await self._index_api.create_index_for_model(req)

        if timeout == -1:
            return IndexModel(resp)
        return await self.__poll_describe_index_until_ready(name, timeout)

    @require_kwargs
    async def create_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: Optional[Union[DeletionProtection, str]] = DeletionProtection.DISABLED,
        tags: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> IndexModel:
        req = PineconeDBControlRequestFactory.create_index_from_backup_request(
            name=name, deletion_protection=deletion_protection, tags=tags
        )
        await self._index_api.create_index_from_backup_operation(
            backup_id=backup_id, create_index_from_backup_request=req
        )
        return await self.__poll_describe_index_until_ready(name, timeout)

    async def __poll_describe_index_until_ready(self, name: str, timeout: Optional[int] = None):
        total_wait_time = 0
        while True:
            description = await self.describe(name=name)
            if description.status.state == "InitializationFailed":
                raise Exception(f"Index {name} failed to initialize.")
            if description.status.ready:
                return description

            if timeout is not None and total_wait_time >= timeout:
                logger.error(
                    f"Index {name} is not ready after {total_wait_time} seconds. Timeout reached."
                )
                link = docslinks["API_DESCRIBE_INDEX"](API_VERSION)
                timeout_msg = f"Index {name} is not ready after {total_wait_time} seconds. Please call describe_index() to confirm index status. See docs at {link}"
                raise TimeoutError(timeout_msg)

            logger.debug(
                f"Waiting for index {name} to be ready. Total wait time {total_wait_time} seconds."
            )

            total_wait_time += 5
            await asyncio.sleep(5)

    @require_kwargs
    async def delete(self, *, name: str, timeout: Optional[int] = None):
        await self._index_api.delete_index(name)

        if timeout == -1:
            return

        if timeout is None:
            while await self.has(name=name):
                await asyncio.sleep(5)
        else:
            while await self.has(name=name) and timeout >= 0:
                await asyncio.sleep(5)
                timeout -= 5
        if timeout and timeout < 0:
            raise (
                TimeoutError(
                    "Please call the list_indexes API ({}) to confirm if index is deleted".format(
                        "https://www.pinecone.io/docs/api/operation/list_indexes/"
                    )
                )
            )

    @require_kwargs
    async def list(self) -> IndexList:
        response = await self._index_api.list_indexes()
        return IndexList(response)

    @require_kwargs
    async def describe(self, *, name: str) -> IndexModel:
        description = await self._index_api.describe_index(name)
        return IndexModel(description)

    @require_kwargs
    async def has(self, *, name: str) -> bool:
        available_indexes = await self.list()
        if name in available_indexes.names():
            return True
        else:
            return False

    @require_kwargs
    async def configure(
        self,
        *,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[Union[PodType, str]] = None,
        deletion_protection: Optional[Union[DeletionProtection, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        description = await self.describe(name=name)

        req = PineconeDBControlRequestFactory.configure_index_request(
            description=description,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
        )
        await self._index_api.configure_index(name, configure_index_request=req)
