from typing import TYPE_CHECKING
import logging

from pinecone.db_control.models import CollectionList
from pinecone.db_control.request_factory import PineconeDBControlRequestFactory
from pinecone.utils import PluginAware, require_kwargs

logger = logging.getLogger(__name__)
""" @private """

if TYPE_CHECKING:
    from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
    from pinecone.config import Config, OpenApiConfiguration


class CollectionResource(PluginAware):
    def __init__(
        self,
        index_api: "ManageIndexesApi",
        config: "Config",
        openapi_config: "OpenApiConfiguration",
        pool_threads: int,
    ):
        self.index_api = index_api
        """ @private """


        self.config = config
        """ @private """

        self._openapi_config = openapi_config
        """ @private """

        self._pool_threads = pool_threads
        """ @private """

        super().__init__()  # Initialize PluginAware

    @require_kwargs
    def create(self, *, name: str, source: str) -> None:
        req = PineconeDBControlRequestFactory.create_collection_request(name=name, source=source)
        self.index_api.create_collection(create_collection_request=req)

    @require_kwargs
    def list(self) -> CollectionList:
        response = self.index_api.list_collections()
        return CollectionList(response)

    @require_kwargs
    def delete(self, *, name: str) -> None:
        self.index_api.delete_collection(name)

    @require_kwargs
    def describe(self, *, name: str):
        return self.index_api.describe_collection(name).to_dict()
