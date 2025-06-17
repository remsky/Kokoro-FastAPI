from typing import Optional, TYPE_CHECKING

from pinecone.db_control.models import RestoreJobModel, RestoreJobList
from pinecone.utils import parse_non_empty_args, require_kwargs, PluginAware

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration
    from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi


class RestoreJobResource(PluginAware):
    def __init__(
        self,
        index_api: "ManageIndexesApi",
        config: "Config",
        openapi_config: "OpenApiConfiguration",
        pool_threads: int,
    ):
        self._index_api = index_api
        """ @private """

        self.config = config
        """ @private """

        self._openapi_config = openapi_config
        """ @private """

        self._pool_threads = pool_threads
        """ @private """

        super().__init__()  # Initialize PluginAware

    @require_kwargs
    def get(self, *, job_id: str) -> RestoreJobModel:
        """
        Get a restore job by ID.

        Args:
            job_id (str): The ID of the restore job to get.

        Returns:
            RestoreJobModel: The restore job.
        """
        job = self._index_api.describe_restore_job(job_id=job_id)
        return RestoreJobModel(job)

    @require_kwargs
    def describe(self, *, job_id: str) -> RestoreJobModel:
        """
        Get a restore job by ID. Alias for get.

        Args:
            job_id (str): The ID of the restore job to get.

        Returns:
            RestoreJobModel: The restore job.
        """
        return self.get(job_id=job_id)

    @require_kwargs
    def list(
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
        jobs = self._index_api.list_restore_jobs(**args)
        return RestoreJobList(jobs)
