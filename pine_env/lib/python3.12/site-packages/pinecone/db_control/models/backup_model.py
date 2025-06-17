import json
from pinecone.core.openapi.db_control.model.backup_model import BackupModel as OpenAPIBackupModel
from pinecone.utils.repr_overrides import custom_serializer


class BackupModel:
    def __init__(self, backup: OpenAPIBackupModel):
        self._backup = backup

    def __getattr__(self, attr):
        return getattr(self._backup, attr)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self):
        return self._backup.to_dict()
