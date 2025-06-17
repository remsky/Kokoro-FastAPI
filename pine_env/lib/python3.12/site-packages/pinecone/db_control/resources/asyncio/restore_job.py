from typing import Optional

from pinecone.core.openapi.db_control.api.manage_indexes_api import AsyncioManageIndexesApi
from pinecone.db_control.models import RestoreJobModel, RestoreJobList
from pinecone.utils import parse_non_empty_args, require_kwargs


class RestoreJobResourceAsyncio:
    def __init__(self, index_api: AsyncioManageIndexesApi):
        self._index_api = index_api
        """ @private """

    @require_kwargs
    async def get(self, *, job_id: str) -> RestoreJobModel:
        """
        Get a restore job by ID.

        Args:
            job_id (str): The ID of the restore job to get.

        Returns:
            RestoreJobModel: The restore job.
        """
        job = await self._index_api.describe_restore_job(job_id=job_id)
        return RestoreJobModel(job)

    @require_kwargs
    async def describe(self, *, job_id: str) -> RestoreJobModel:
        """
        Get a restore job by ID. Alias for get.

        Args:
            job_id (str): The ID of the restore job to get.

        Returns:
            RestoreJobModel: The restore job.
        """
        return await self.get(job_id=job_id)

    @require_kwargs
    async def list(
        self, *, limit: Optional[int] = 10, pagination_token: Optional[str] = None
    ) -> RestoreJobList:
        """
        List all restore jobs.

        Args:
            limit (int): The maximum number of restore jobs to return.
            pagination_token (str): The pagination token to use for the next page of restore jobs.

        Returns:
            List[RestoreJobModel]: The list of restore jobs.
        """
        args = parse_non_empty_args([("limit", limit), ("pagination_token", pagination_token)])
        jobs = await self._index_api.list_restore_jobs(**args)
        return RestoreJobList(jobs)
