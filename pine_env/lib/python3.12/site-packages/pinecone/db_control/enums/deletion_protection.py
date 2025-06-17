from enum import Enum


class DeletionProtection(Enum):
    """The DeletionProtection setting of an index indicates whether the index
    can be  the index cannot be deleted using the delete_index() method.

    If disabled, the index can be deleted. If enabled, calling delete_index()
    will raise an error.

    This setting can be changed using the configure_index() method.
    """

    ENABLED = "enabled"
    DISABLED = "disabled"
