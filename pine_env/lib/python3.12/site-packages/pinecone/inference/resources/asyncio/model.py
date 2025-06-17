from typing import TYPE_CHECKING, Optional
from pinecone.utils import require_kwargs, parse_non_empty_args
from ...models import ModelInfoList, ModelInfo


if TYPE_CHECKING:
    from pinecone.core.openapi.inference.api.inference_api import AsyncioInferenceApi


class ModelAsyncio:
    def __init__(self, inference_api: "AsyncioInferenceApi") -> None:
        self.__inference_api = inference_api
        """ @private """

        super().__init__()  # Initialize PluginAware

    @require_kwargs
    async def list(
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
        model_list = await self.__inference_api.list_models(**args)
        return ModelInfoList(model_list)

    @require_kwargs
    async def get(self, model_name: str) -> ModelInfo:
        """
        Get a specific model by name.

        :param model_name: The name of the model to get.
        :type model_name: str, required

        :return: A model.
        """
        model_info = await self.__inference_api.get_model(model_name=model_name)
        return ModelInfo(model_info)
