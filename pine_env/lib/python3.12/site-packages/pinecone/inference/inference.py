import logging
import warnings
from typing import Optional, Dict, List, Union, Any, TYPE_CHECKING

from pinecone.openapi_support import ApiClient
from pinecone.core.openapi.inference.apis import InferenceApi
from .models import EmbeddingsList, RerankResult
from pinecone.core.openapi.inference import API_VERSION
from pinecone.utils import setup_openapi_client, PluginAware
from pinecone.utils import require_kwargs

from .inference_request_builder import (
    InferenceRequestBuilder,
    EmbedModel as EmbedModelEnum,
    RerankModel as RerankModelEnum,
)

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from .resources.sync.model import Model as ModelResource
    from .models import ModelInfo, ModelInfoList

logger = logging.getLogger(__name__)
""" @private """


class Inference(PluginAware):
    """
    The `Inference` class configures and uses the Pinecone Inference API to generate embeddings and
    rank documents.

    It is generally not instantiated directly, but rather accessed through a parent `Pinecone` client
    object that is responsible for managing shared configurations.

    ```python
    from pinecone import Pinecone

    pc = Pinecone()
    embeddings = pc.inference.embed(
        model="text-embedding-3-small",
        inputs=["Hello, world!"],
        parameters={"input_type": "passage", "truncate": "END"}
    )
    ```

    :param config: A `pinecone.config.Config` object, configured and built in the Pinecone class.
    :type config: `pinecone.config.Config`, required
    """

    EmbedModel = EmbedModelEnum
    RerankModel = RerankModelEnum

    def __init__(
        self,
        config: "Config",
        openapi_config: "OpenApiConfiguration",
        pool_threads: int = 1,
        **kwargs,
    ) -> None:
        self._config = config
        """ @private """

        self._openapi_config = openapi_config
        """ @private """

        self._pool_threads = pool_threads
        """ @private """

        self.__inference_api = setup_openapi_client(
            api_client_klass=ApiClient,
            api_klass=InferenceApi,
            config=config,
            openapi_config=openapi_config,
            pool_threads=self._pool_threads,
            api_version=API_VERSION,
        )

        self._model: Optional["ModelResource"] = None  # Lazy initialization
        """ @private """

        super().__init__()  # Initialize PluginAware

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
    def model(self) -> "ModelResource":
        """
        Model is a resource that describes models available in the Pinecone Inference API.

        Curently you can get or list models.

        ```python
        pc = Pinecone()

        # List all models
        models = pc.inference.model.list()

        # List models, with model type filtering
        models = pc.inference.model.list(type="embed")
        models = pc.inference.model.list(type="rerank")

        # List models, with vector type filtering
        models = pc.inference.model.list(vector_type="dense")
        models = pc.inference.model.list(vector_type="sparse")

        # List models, with both type and vector type filtering
        models = pc.inference.model.list(type="rerank", vector_type="dense")

        # Get details on a specific model
        model = pc.inference.model.get("text-embedding-3-small")
        ```
        """
        if self._model is None:
            from .resources.sync.model import Model as ModelResource

            self._model = ModelResource(
                inference_api=self.__inference_api,
                config=self._config,
                openapi_config=self._openapi_config,
                pool_threads=self._pool_threads,
            )
        return self._model

    def embed(
        self,
        model: Union[EmbedModelEnum, str],
        inputs: Union[str, List[Dict], List[str]],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingsList:
        """
        Generates embeddings for the provided inputs using the specified model and (optional) parameters.

        :param model: The model to use for generating embeddings.
        :type model: str, required

        :param inputs: A list of items to generate embeddings for.
        :type inputs: list, required

        :param parameters: A dictionary of parameters to use when generating embeddings.
        :type parameters: dict, optional

        :return: EmbeddingsList object with keys `data`, `model`, and `usage`. The `data` key contains a list of
        `n` embeddings, where `n` = len(inputs) and type(n) = Embedding. Precision of returned embeddings is either
        float16 or float32, with float32 being the default. `model` key is the model used to generate the embeddings.
        `usage` key contains the total number of tokens used at request-time.

        Example:
        >>> inputs = ["Who created the first computer?"]
        >>> outputs = pc.inference.embed(model="multilingual-e5-large", inputs=inputs, parameters={"input_type": "passage", "truncate": "END"})
        >>> print(outputs)
        EmbeddingsList(
            model='multilingual-e5-large',
            data=[
                {'values': [0.1, ...., 0.2]},
              ],
            usage={'total_tokens': 6}
        )
        """
        request_body = InferenceRequestBuilder.embed_request(
            model=model, inputs=inputs, parameters=parameters
        )
        resp = self.__inference_api.embed(embed_request=request_body)
        return EmbeddingsList(resp)

    def rerank(
        self,
        model: Union[RerankModelEnum, str],
        query: str,
        documents: Union[List[str], List[Dict[str, Any]]],
        rank_fields: List[str] = ["text"],
        return_documents: bool = True,
        top_n: Optional[int] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> RerankResult:
        """
        Rerank documents with associated relevance scores that represent the relevance of each document
        to the provided query using the specified model.

        :param model: The model to use for reranking.
        :type model: str, required

        :param query: The query to compare with documents.
        :type query: str, required

        :param documents: A list of documents or strings to rank.
        :type documents: list, required

        :param rank_fields: A list of document fields to use for ranking. Defaults to ["text"].
        :type rank_fields: list, optional

        :param return_documents: Whether to include the documents in the response. Defaults to True.
        :type return_documents: bool, optional

        :param top_n: How many documents to return. Defaults to len(documents).
        :type top_n: int, optional

        :param parameters: A dictionary of parameters to use when ranking documents.
        :type parameters: dict, optional

        :return: RerankResult object with keys `data` and `usage`. The `data` key contains a list of
        `n` documents, where `n` = `top_n` and type(n) = Document. The documents are sorted in order of
        relevance, with the first being the most relevant. The `index` field can be used to locate the document
        relative to the list of documents specified in the request. Each document contains a `score` key
        representing how close the document relates to the query.

        Example:
        >>> result = pc.inference.rerank(
                model="bge-reranker-v2-m3",
                query="Tell me about tech companies",
                documents=[
                    "Apple is a popular fruit known for its sweetness and crisp texture.",
                    "Software is still eating the world.",
                    "Many people enjoy eating apples as a healthy snack.",
                    "Acme Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
                    "An apple a day keeps the doctor away, as the saying goes.",
                ],
                top_n=2,
                return_documents=True,
            )
        >>> print(result)
        RerankResult(
          model='bge-reranker-v2-m3',
          data=[
            { index=3, score=0.020980744,
              document={text="Acme Inc. has rev..."} },
            { index=1, score=0.00034015716,
              document={text="Software is still..."} }
          ],
          usage={'rerank_units': 1}
        )
        """
        rerank_request = InferenceRequestBuilder.rerank(
            model=model,
            query=query,
            documents=documents,
            rank_fields=rank_fields,
            return_documents=return_documents,
            top_n=top_n,
            parameters=parameters,
        )
        resp = self.__inference_api.rerank(rerank_request=rerank_request)
        return RerankResult(resp)

    @require_kwargs
    def list_models(
        self, *, type: Optional[str] = None, vector_type: Optional[str] = None
    ) -> "ModelInfoList":
        """
         List all available models.


         ```python
         pc = Pinecone()

         # List all models
         models = pc.inference.list_models()

         # List models, with model type filtering
         models = pc.inference.list_models(type="embed")
         models = pc.inference.list_models(type="rerank")

         # List models, with vector type filtering
         models = pc.inference.list_models(vector_type="dense")
         models = pc.inference.list_models(vector_type="sparse")

         # List models, with both type and vector type filtering
         models = pc.inference.list_models(type="rerank", vector_type="dense")
        ```

         :param type: The type of model to list. Either "embed" or "rerank".
         :type type: str, optional

         :param vector_type: The type of vector to list. Either "dense" or "sparse".
         :type vector_type: str, optional

         :return: A list of models.
        """
        return self.model.list(type=type, vector_type=vector_type)

    @require_kwargs
    def get_model(self, model_name: str) -> "ModelInfo":
        """
        Get details on a specific model.

        ```python
        pc = Pinecone()

        model = pc.inference.get_model(model_name="text-embedding-3-small")
        ```

        :param model_name: The name of the model to get details on.
        :type model_name: str, required

        :return: A ModelInfo object.
        """
        return self.model.get(model_name=model_name)
