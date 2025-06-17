from abc import ABC, abstractmethod


class DictMixin(ABC):
    def __getitem__(self, key):
        data = self.to_dict()
        if key not in data:
            raise KeyError(f"Key '{key}' not found in the object.")
        return data[key]

    def __contains__(self, key):
        return key in self.to_dict()

    def __len__(self):
        return len(self.to_dict())

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def get(self, key, default=None):
        return self.to_dict().get(key, default)

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())

    @abstractmethod
    def to_dict(self):
        raise NotImplementedError
