from pinecone import Pinecone
from pinecone.config import ConfigBuilder
from .index_grpc import GRPCIndex


class PineconeGRPC(Pinecone):
    """
    An alternative version of the Pinecone client that uses gRPC instead of HTTP for
    data operations.

    ### Installing the gRPC client

    You must install extra dependencies in order to install the GRPC client.

    #### Installing with pip

    ```bash
    # Install the latest version
    pip3 install pinecone[grpc]

    # Install a specific version
    pip3 install "pinecone[grpc]"==3.0.0
    ```

    #### Installing with poetry

    ```bash
    # Install the latest version
    poetry add pinecone --extras grpc

    # Install a specific version
    poetry add pinecone==3.0.0 --extras grpc
    ```

    ### Using the gRPC client

    ```python
    import os
    from pinecone.grpc import PineconeGRPC

    client = PineconeGRPC(api_key=os.environ.get("PINECONE_API_KEY"))

    # From this point on, usage is identical to the HTTP client.
    index = client.Index("my-index", host=os.environ("PINECONE_INDEX_HOST"))
    index.query(...)
    ```

    """

    def Index(self, name: str = "", host: str = "", **kwargs):
        """
        Target an index for data operations.

        ### Target an index by host url

        In production situations, you want to uspert or query your data as quickly
        as possible. If you know in advance the host url of your index, you can
        eliminate a round trip to the Pinecone control plane by specifying the
        host of the index.

        ```python
        import os
        from pinecone.grpc import PineconeGRPC

        api_key = os.environ.get("PINECONE_API_KEY")
        index_host = os.environ.get("PINECONE_INDEX_HOST")

        pc = PineconeGRPC(api_key=api_key)
        index = pc.Index(host=index_host)

        # Now you're ready to perform data operations
        index.query(vector=[...], top_k=10)
        ```

        To find your host url, you can use the Pinecone control plane to describe
        the index. The host url is returned in the response. Or, alternatively, the
        host is displayed in the Pinecone web console.

        ```python
        import os
        from pinecone import Pinecone

        pc = Pinecone(
            api_key=os.environ.get("PINECONE_API_KEY")
        )

        host = pc.describe_index('index-name').host
        ```

        ### Target an index by name (not recommended for production)

        For more casual usage, such as when you are playing and exploring with Pinecone
        in a notebook setting, you can also target an index by name. If you use this
        approach, the client may need to perform an extra call to the Pinecone control
        plane to get the host url on your behalf to get the index host.

        The client will cache the index host for future use whenever it is seen, so you
        will only incur the overhead of only one call. But this approach is not
        recommended for production usage.

        ```python
        import os
        from pinecone import ServerlessSpec
        from pinecone.grpc import PineconeGRPC

        api_key = os.environ.get("PINECONE_API_KEY")

        pc = PineconeGRPC(api_key=api_key)
        pc.create_index(
            name='my-index',
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )
        index = pc.Index('my-index')

        # Now you're ready to perform data operations
        index.query(vector=[...], top_k=10)
        ```
        """
        if name == "" and host == "":
            raise ValueError("Either name or host must be specified")

        # Use host if it is provided, otherwise get host from describe_index
        index_host = host or self.db.index._get_host(name)

        pt = kwargs.pop("pool_threads", None) or self._pool_threads

        config = ConfigBuilder.build(
            api_key=self._config.api_key,
            host=index_host,
            source_tag=self._config.source_tag,
            proxy_url=self._config.proxy_url,
            ssl_ca_certs=self._config.ssl_ca_certs,
        )
        return GRPCIndex(index_name=name, config=config, pool_threads=pt, **kwargs)
