from dataclasses import asdict, dataclass

from pinecone_plugins.assistant.models.core.dict_mixin import DictMixin


@dataclass
class BaseDataclass(DictMixin):
    def to_dict(self):
        return asdict(self)
