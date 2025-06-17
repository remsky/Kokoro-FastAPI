import json
from typing import List
from pinecone.core.openapi.inference.model.model_info_list import (
    ModelInfoList as OpenAPIModelInfoList,
)
from .model_info import ModelInfo
from pinecone.utils.repr_overrides import custom_serializer


class ModelInfoList:
    """
    A list of model information.
    """

    def __init__(self, model_info_list: OpenAPIModelInfoList):
        self._model_info_list = model_info_list
        self._models = [ModelInfo(model_info) for model_info in model_info_list.models]

    def names(self) -> List[str]:
        return [i.name for i in self._models]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._models[key]
        elif key == "models":
            # Return mapped models
            return self._models
        else:
            # any other keys added in the future
            return self._model_info_list[key]

    def __getattr__(self, attr):
        if attr == "models":
            return self._models
        else:
            # any other keys added in the future
            return getattr(self._model_info_list, attr)

    def __len__(self):
        return len(self._models)

    def __iter__(self):
        return iter(self._models)

    def __str__(self):
        return str(self._models)

    def __repr__(self):
        raw_dict = self._model_info_list.to_dict()
        raw_dict["models"] = [i.to_dict() for i in self._models]

        # Remove keys with value None
        for key, value in list(raw_dict.items()):
            if value is None:
                del raw_dict[key]

        return json.dumps(raw_dict, indent=4, default=custom_serializer)
