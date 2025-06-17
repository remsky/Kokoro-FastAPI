import json
from pinecone.core.openapi.db_control.model.index_list import IndexList as OpenAPIIndexList
from .index_model import IndexModel
from typing import List


class IndexList:
    def __init__(self, index_list: OpenAPIIndexList):
        self.index_list = index_list
        self.indexes = [IndexModel(i) for i in self.index_list.indexes]
        self.current = 0

    def names(self) -> List[str]:
        return [i.name for i in self.indexes]

    def __getitem__(self, key):
        return self.indexes[key]

    def __len__(self):
        return len(self.indexes)

    def __iter__(self):
        return iter(self.indexes)

    def __str__(self):
        return str(self.indexes)

    def __repr__(self):
        return json.dumps([i.to_dict() for i in self.indexes], indent=4)

    def __getattr__(self, attr):
        return getattr(self.index_list, attr)
