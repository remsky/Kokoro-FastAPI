def _build_class_migration_message(method_name: str, example: str):
    return f"""{method_name} is no longer a top-level attribute of the pinecone package.

To use {method_name}, please create a client instance and call the method there instead.

Example:
{example}
"""


def init(*args, **kwargs):
    example = """
    import os
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    # Now do stuff
    if 'my_index' not in pc.list_indexes().names():
        pc.create_index(
            name='my_index', 
            dimension=1536, 
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )
"""
    msg = f"""init is no longer a top-level attribute of the pinecone package.

Please create an instance of the Pinecone class instead.

Example:
{example}
"""
    raise AttributeError(msg)


def list_indexes(*args, **kwargs):
    example = """
    from pinecone import Pinecone
    
    pc = Pinecone(api_key='YOUR_API_KEY')

    index_name = "quickstart" # or your index name

    if index_name not in pc.list_indexes().names():
        # do something
"""
    raise AttributeError(_build_class_migration_message("list_indexes", example))


def describe_index(*args, **kwargs):
    example = """
    from pinecone import Pinecone
    
    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.describe_index('my_index')
"""
    raise AttributeError(_build_class_migration_message("describe_index", example))


def create_index(*args, **kwargs):
    example = """
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.create_index(
        name='my-index',
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
"""
    raise AttributeError(_build_class_migration_message("create_index", example))


def delete_index(*args, **kwargs):
    example = """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.delete_index('my_index')
"""
    raise AttributeError(_build_class_migration_message("delete_index", example))


def scale_index(*args, **kwargs):
    example = """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.configure_index('my_index', replicas=2)
"""

    msg = f"""scale_index is no longer a top-level attribute of the pinecone package.

Please create a client instance and call the configure_index method instead.

Example:
{example}
"""
    raise AttributeError(msg)


def create_collection(*args, **kwargs):
    example = """
    from pinecone import Pinecone
    
    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.create_collection(name='my_collection', source='my_index')
"""
    raise AttributeError(_build_class_migration_message("create_collection", example))


def list_collections(*args, **kwargs):
    example = """
    from pinecone import Pinecone

    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.list_collections()
"""
    raise AttributeError(_build_class_migration_message("list_collections", example))


def delete_collection(*args, **kwargs):
    example = """
    from pinecone import Pinecone
    
    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.delete_collection('my_collection')
"""
    raise AttributeError(_build_class_migration_message("delete_collection", example))


def describe_collection(*args, **kwargs):
    example = """
    from pinecone import Pinecone
    
    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.describe_collection('my_collection')
"""
    raise AttributeError(_build_class_migration_message("describe_collection", example))


def configure_index(*args, **kwargs):
    example = """
    from pinecone import Pinecone
    
    pc = Pinecone(api_key='YOUR_API_KEY')
    pc.configure_index('my_index', replicas=2)
"""
    raise AttributeError(_build_class_migration_message("configure_index", example))
