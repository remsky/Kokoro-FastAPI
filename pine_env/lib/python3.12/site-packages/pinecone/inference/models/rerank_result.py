from pinecone.core.openapi.inference.models import RerankResult as OpenAPIRerankResult


class RerankResult:
    """
    A wrapper around OpenAPIRerankResult.
    """

    def __init__(self, rerank_result: OpenAPIRerankResult):
        self.rerank_result = rerank_result

    def __str__(self):
        return str(self.rerank_result)

    def __repr__(self):
        return repr(self.rerank_result)

    def __getattr__(self, attr):
        return getattr(self.rerank_result, attr)
