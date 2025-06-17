import json
from pinecone.core.openapi.db_control.model.restore_job_model import (
    RestoreJobModel as OpenAPIRestoreJobModel,
)
from pinecone.utils.repr_overrides import custom_serializer


class RestoreJobModel:
    def __init__(self, restore_job: OpenAPIRestoreJobModel):
        self.restore_job = restore_job

    def __str__(self):
        return str(self.restore_job)

    def __getattr__(self, attr):
        return getattr(self.restore_job, attr)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=4, default=custom_serializer)

    def to_dict(self):
        return self.restore_job.to_dict()
