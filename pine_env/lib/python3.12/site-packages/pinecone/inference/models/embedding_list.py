from pinecone.core.openapi.inference.models import EmbeddingsList as OpenAPIEmbeddingsList


class EmbeddingsList:
    """
    A list of embeddings.
    """

    def __init__(self, embeddings_list: OpenAPIEmbeddingsList):
        self.embeddings_list = embeddings_list
        self.current = 0

    def __getitem__(self, index):
        return self.embeddings_list.get("data")[index]

    def __len__(self):
        return len(self.embeddings_list.get("data"))

    def __iter__(self):
        return iter(self.embeddings_list.get("data"))

    def __str__(self):
        return str(self.embeddings_list)

    def __repr__(self):
        return repr(self.embeddings_list)

    def __getattr__(self, attr):
        return getattr(self.embeddings_list, attr)
