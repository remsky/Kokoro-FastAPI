import json
from pinecone.core.openapi.db_control.model.restore_job_list import (
    RestoreJobList as OpenAPIRestoreJobList,
)
from .restore_job_model import RestoreJobModel

from datetime import datetime


def custom_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)


class RestoreJobList:
    def __init__(self, restore_job_list: OpenAPIRestoreJobList):
        self._restore_job_list = restore_job_list
        self._restore_jobs = [RestoreJobModel(r) for r in self._restore_job_list.data]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._restore_jobs[key]
        elif key == "data":
            return self._restore_jobs
        else:
            # pagination and any other keys added in the future
            return self._restore_job_list[key]

    def __getattr__(self, attr):
        if attr == "data":
            return self._restore_jobs
        else:
            # pagination and any other keys added in the future
            return getattr(self._restore_job_list, attr)

    def __len__(self):
        return len(self._restore_jobs)

    def __iter__(self):
        return iter(self._restore_jobs)

    def __str__(self):
        return str(self._restore_jobs)

    def __repr__(self):
        return json.dumps(
            [i.to_dict() for i in self._restore_jobs], indent=4, default=custom_serializer
        )
