from typing import Optional, TYPE_CHECKING

from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi
from pinecone.core.openapi.db_control.model.create_backup_request import CreateBackupRequest
from pinecone.db_control.models import BackupModel, BackupList
from pinecone.utils import parse_non_empty_args, require_kwargs, PluginAware

if TYPE_CHECKING:
    from pinecone.config import Config, OpenApiConfiguration


class BackupResource(PluginAware):
    def __init__(
        self,
        index_api: ManageIndexesApi,
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
    def list(
        self,
        *,
        index_name: Optional[str] = None,
        limit: Optional[int] = 10,
        pagination_token: Optional[str] = None,
    ) -> BackupList:
        """
        List backups for an index or for the project.

        Args:
            index_name (str): The name of the index to list backups for. If not provided, list all backups for the project.
            limit (int): The maximum number of backups to return.
            pagination_token (str): The pagination token to use for the next page of backups.
        """
        if index_name is not None:
            args = parse_non_empty_args(
                [
                    ("index_name", index_name),
                    ("limit", limit),
                    ("pagination_token", pagination_token),
                ]
            )
            return BackupList(self._index_api.list_index_backups(**args))
        else:
            args = parse_non_empty_args([("limit", limit), ("pagination_token", pagination_token)])
            return BackupList(self._index_api.list_project_backups(**args))

    @require_kwargs
    def create(self, *, index_name: str, backup_name: str, description: str = "") -> BackupModel:
        """
        Create a backup for an index.

        Args:
            index_name (str): The name of the index to create a backup for.
            backup_name (str): The name of the backup to create.
            description (str): The description of the backup.

        Returns:
            BackupModel: The created backup.
        """
        req = CreateBackupRequest(name=backup_name, description=description)
        return BackupModel(
            self._index_api.create_backup(index_name=index_name, create_backup_request=req)
        )

    @require_kwargs
    def describe(self, *, backup_id: str) -> BackupModel:
        """
        Describe a backup.

        Args:
            backup_id (str): The ID of the backup to describe.

        Returns:
            BackupModel: The described backup.
        """
        return BackupModel(self._index_api.describe_backup(backup_id=backup_id))

    @require_kwargs
    def get(self, *, backup_id: str) -> BackupModel:
        """Alias for describe"""
        return self.describe(backup_id=backup_id)

    @require_kwargs
    def delete(self, *, backup_id: str) -> None:
        """
        Delete a backup.

        Args:
            backup_id (str): The ID of the backup to delete.
        """
        return self._index_api.delete_backup(backup_id=backup_id)
