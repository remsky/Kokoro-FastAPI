import json
from pinecone.core.openapi.db_control.model.collection_list import (
    CollectionList as OpenAPICollectionList,
)


class CollectionList:
    """
    A list of collections.
    """

    def __init__(self, collection_list: OpenAPICollectionList):
        self.collection_list = collection_list
        self.current = 0

    def names(self):
        return [i["name"] for i in self.collection_list.collections]

    def __getitem__(self, key):
        return self.collection_list.collections[key]

    def __len__(self):
        return len(self.collection_list.collections)

    def __iter__(self):
        return iter(self.collection_list.collections)

    def __str__(self):
        return str(self.collection_list)

    def __repr__(self):
        return json.dumps([c.to_dict() for c in self.collection_list.collections], indent=4)

    def __getattr__(self, attr):
        return getattr(self.collection_list, attr)
