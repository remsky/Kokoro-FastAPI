from pinecone.utils.repr_overrides import install_json_repr_override
from pinecone.core.openapi.db_control.model.collection_model import CollectionModel


def install_repr_overrides():
    """
    The generator code uses pprint.pformat to format the repr output
    which looks really poor when printing a list of large objects
    in a notebook setting. We override it here for a few select models
    instead of modifying the generator code because the more compact output
    from pprint.pformat seems better for data plane objects such as lists of
    query results.
    """
    for model in [CollectionModel]:
        install_json_repr_override(model)
