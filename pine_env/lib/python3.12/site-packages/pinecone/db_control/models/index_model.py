from pinecone.core.openapi.db_control.model.index_model import IndexModel as OpenAPIIndexModel
import json
from pinecone.utils.repr_overrides import custom_serializer


class IndexModel:
    def __init__(self, index: OpenAPIIndexModel):
        self.index = index
        self.deletion_protection = index.deletion_protection.value

    def __str__(self):
        return str(self.index)

    def __getattr__(self, attr):
        return getattr(self.index, attr)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self):
        return self.index.to_dict()
