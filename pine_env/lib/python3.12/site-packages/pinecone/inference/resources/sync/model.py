from typing import TYPE_CHECKING, Optional
from pinecone.utils import PluginAware, require_kwargs, parse_non_empty_args
from ...models import ModelInfoList, ModelInfo


if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from pinecone.core.openapi.inference.api.inference_api import InferenceApi


class Model(PluginAware):
    def __init__(
        self,
        inference_api: "InferenceApi",
        config: "Config",
        openapi_config: "OpenApiConfiguration",
        pool_threads: int = 1,
        **kwargs,
    ) -> None:
        self._config = config
        """ @private """

        self._openapi_config = openapi_config
        """ @private """

        self._pool_threads = kwargs.get("pool_threads", 1)
        """ @private """

        self.__inference_api = inference_api
        """ @private """

        super().__init__()  # Initialize PluginAware

    @property
    def config(self) -> "Config":
        """@private"""
        # The config property is considered private, but the name cannot be changed to include underscore
        # without breaking compatibility with plugins in the wild.
        return self._config

    @require_kwargs
    def list(
        self, *, type: Optional[str] = None, vector_type: Optional[str] = None
    ) -> ModelInfoList:
        """
        List all available models.

        :param type: The type of model to list. Either "embed" or "rerank".
        :type type: str, optional

        :param vector_type: The type of vector to list. Either "dense" or "sparse".
        :type vector_type: str, optional

        :return: A list of models.
        """
        args = parse_non_empty_args([("type", type), ("vector_type", vector_type)])
        return ModelInfoList(self.__inference_api.list_models(**args))

    @require_kwargs
    def get(self, model_name: str) -> ModelInfo:
        """
        Get a specific model by name.

        :param model_name: The name of the model to get.
        :type model_name: str, required

        :return: A model.
        """
        return ModelInfo(self.__inference_api.get_model(model_name=model_name))
