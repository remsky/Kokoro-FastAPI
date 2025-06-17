from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pinecone.inference import RerankModel


@dataclass
class SearchRerank:
    """
    SearchRerank represents a rerank request when searching within a specific namespace.
    """

    model: str
    """
    The name of the [reranking model](https://docs.pinecone.io/guides/inference/understanding-inference#reranking-models) to use.
    Required.
    """

    rank_fields: List[str]
    """
    The fields to use for reranking.
    Required.
    """

    top_n: Optional[int] = None
    """
    The number of top results to return after reranking. Defaults to top_k.
    Optional.
    """

    parameters: Optional[Dict[str, Any]] = None
    """
    Additional model-specific parameters. Refer to the [model guide](https://docs.pinecone.io/guides/inference/understanding-inference#models)
    for available model parameters.
    Optional.
    """

    query: Optional[str] = None
    """
    The query to rerank documents against. If a specific rerank query is specified, it overwrites
    the query input that was provided at the top level.
    """

    def __post_init__(self):
        """
        Converts `model` to a string if an instance of `RerankEnum` is provided.
        """
        if isinstance(self.model, RerankModel):
            self.model = self.model.value  # Convert Enum to string

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns the SearchRerank as a dictionary.
        """
        d = {
            "model": self.model,
            "rank_fields": self.rank_fields,
            "top_n": self.top_n,
            "parameters": self.parameters,
            "query": self.query,
        }
        return {k: v for k, v in d.items() if v is not None}
