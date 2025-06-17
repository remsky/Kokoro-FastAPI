import time
import logging
from typing import Optional, Dict, Union, TYPE_CHECKING

from pinecone.db_control.index_host_store import IndexHostStore

from pinecone.db_control.models import IndexModel, IndexList, IndexEmbed
from pinecone.utils import docslinks, require_kwargs, PluginAware

from pinecone.db_control.types import CreateIndexForModelEmbedTypedDict
from pinecone.db_control.request_factory import PineconeDBControlRequestFactory
from pinecone.core.openapi.db_control import API_VERSION

logger = logging.getLogger(__name__)
""" @private """

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
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
    from pinecone.db_control.models import ServerlessSpec, PodSpec, ByocSpec, IndexEmbed


class IndexResource(PluginAware):
    def __init__(
        self,
        index_api: "ManageIndexesApi",
        config: "Config",
        openapi_config: "OpenApiConfiguration",
        pool_threads: int,
    ):
        self._index_api = index_api
        """ @private """

        self.config = config
        """ @private """

        self._openapi_config = openapi_config
        """ @private """

        self._pool_threads = pool_threads
        """ @private """

        self._index_host_store = IndexHostStore()
        """ @private """

        super().__init__()  # Initialize PluginAware

    @require_kwargs
    def create(
        self,
        *,
        name: str,
        spec: Union[Dict, "ServerlessSpec", "PodSpec", "ByocSpec"],
        dimension: Optional[int] = None,
        metric: Optional[Union["Metric", str]] = "cosine",
        timeout: Optional[int] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        vector_type: Optional[Union["VectorType", str]] = "dense",
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
        resp = self._index_api.create_index(create_index_request=req)

        if timeout == -1:
            return IndexModel(resp)
        return self.__poll_describe_index_until_ready(name, timeout)

    @require_kwargs
    def create_for_model(
        self,
        *,
        name: str,
        cloud: Union["CloudProvider", str],
        region: Union["AwsRegion", "GcpRegion", "AzureRegion", str],
        embed: Union["IndexEmbed", "CreateIndexForModelEmbedTypedDict"],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
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
        resp = self._index_api.create_index_for_model(req)

        if timeout == -1:
            return IndexModel(resp)
        return self.__poll_describe_index_until_ready(name, timeout)

    @require_kwargs
    def create_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        tags: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> IndexModel:
        """
        Create an index from a backup.

        Args:
            name (str): The name of the index to create.
            backup_id (str): The ID of the backup to create the index from.
            deletion_protection (DeletionProtection): The deletion protection to use for the index.
            tags (Dict[str, str]): The tags to use for the index.
            timeout (int): The number of seconds to wait for the index to be ready. If -1, the function will return without polling for the index status to be ready. If None, the function will poll indefinitely for the index to be ready.

        Returns:
            IndexModel: The created index.
        """
        req = PineconeDBControlRequestFactory.create_index_from_backup_request(
            name=name, deletion_protection=deletion_protection, tags=tags
        )
        resp = self._index_api.create_index_from_backup_operation(
            backup_id=backup_id, create_index_from_backup_request=req
        )
        logger.info(f"Creating index from backup. Response: {resp}")

        if timeout == -1:
            return self.describe(name=name)
        return self.__poll_describe_index_until_ready(name, timeout)

    def __poll_describe_index_until_ready(self, name: str, timeout: Optional[int] = None):
        total_wait_time = 0
        while True:
            description = self.describe(name=name)
            if description.status.state == "InitializationFailed":
                raise Exception(
                    f"Index {name} failed to initialize. The index status is {description.status.state}."
                )
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
            time.sleep(5)

    @require_kwargs
    def delete(self, *, name: str, timeout: Optional[int] = None) -> None:
        self._index_api.delete_index(name)
        self._index_host_store.delete_host(self.config, name)

        if timeout == -1:
            return

        if timeout is None:
            while self.has(name=name):
                time.sleep(5)
        else:
            while self.has(name=name) and timeout >= 0:
                time.sleep(5)
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
    def list(self) -> IndexList:
        response = self._index_api.list_indexes()
        return IndexList(response)

    @require_kwargs
    def describe(self, *, name: str) -> IndexModel:
        api_instance = self._index_api
        description = api_instance.describe_index(name)
        host = description.host
        self._index_host_store.set_host(self.config, name, host)

        return IndexModel(description)

    @require_kwargs
    def has(self, *, name: str) -> bool:
        if name in self.list().names():
            return True
        else:
            return False

    @require_kwargs
    def configure(
        self,
        *,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[Union["PodType", str]] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        api_instance = self._index_api
        description = self.describe(name=name)

        req = PineconeDBControlRequestFactory.configure_index_request(
            description=description,
            replicas=replicas,
            pod_type=pod_type,
            deletion_protection=deletion_protection,
            tags=tags,
        )
        api_instance.configure_index(name, configure_index_request=req)

    def _get_host(self, name: str) -> str:
        """@private"""
        return self._index_host_store.get_host(
            api=self._index_api, config=self.config, index_name=name
        )
