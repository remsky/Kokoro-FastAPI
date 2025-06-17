from abc import ABC, abstractmethod

from typing import Optional, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pinecone.config import Config

    from pinecone.core.openapi.db_control.api.manage_indexes_api import ManageIndexesApi

    from pinecone.db_control.models import (
        ServerlessSpec,
        PodSpec,
        ByocSpec,
        IndexList,
        CollectionList,
        IndexModel,
        IndexEmbed,
        BackupModel,
        BackupList,
        RestoreJobModel,
        RestoreJobList,
    )
    from pinecone.db_control.enums import (
        Metric,
        VectorType,
        DeletionProtection,
        PodType,
        CloudProvider,
        AwsRegion,
        GcpRegion,
        AzureRegion,
    )
    from pinecone.db_control.types import CreateIndexForModelEmbedTypedDict


class PineconeAsyncioDBControlInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        proxy_url: Optional[str] = None,
        proxy_headers: Optional[Dict[str, str]] = None,
        ssl_ca_certs: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        config: Optional["Config"] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        pool_threads: Optional[int] = 1,
        index_api: Optional["ManageIndexesApi"] = None,
        **kwargs,
    ):
        """
        The `PineconeAsyncio` class is the main entry point for interacting with Pinecone using asyncio.
        It is used to create, delete, and manage your indexes and collections. Except for needing to use
        `async with` when instantiating the client and `await` when calling its methods, the functionality
        provided by this class is extremely similar to the functionality of the `Pinecone` class.

        :param api_key: The API key to use for authentication. If not passed via kwarg, the API key will be read from the environment variable `PINECONE_API_KEY`.
        :type api_key: str, optional
        :param host: The control plane host to connect to.
        :type host: str, optional
        :param proxy_url: The URL of the proxy to use for the connection. Default: `None`
        :type proxy_url: str, optional
        :param proxy_headers: Additional headers to pass to the proxy. Use this if your proxy setup requires authentication. Default: `{}`
        :type proxy_headers: Dict[str, str], optional
        :param ssl_ca_certs: The path to the SSL CA certificate bundle to use for the connection. This path should point to a file in PEM format. Default: `None`
        :type ssl_ca_certs: str, optional
        :param ssl_verify: SSL verification is performed by default, but can be disabled using the boolean flag. Default: `True`
        :type ssl_verify: bool, optional
        :param config: A `pinecone.config.Config` object. If passed, the `api_key` and `host` parameters will be ignored.
        :type config: pinecone.config.Config, optional
        :param additional_headers: Additional headers to pass to the API. Default: `{}`
        :type additional_headers: Dict[str, str], optional


        ### Managing the async context

        The `PineconeAsyncio` class relies on an underlying `aiohttp` `ClientSession` to make asynchronous HTTP requests. To ensure that the session is properly closed, you
        should use the `async with` syntax when creating a `PineconeAsyncio` object. This will ensure that the session is properly closed when the context is exited.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio(api_key='YOUR_API_KEY') as pc:
                # Do async things
                index_list = await pc.list_indexes()

        asyncio.run(main())
        ```

        As an alternative, if you prefer to avoid code with a nested appearance and are willing to manage cleanup yourself, you can await the `close()` method to close the session when you are done.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            pc = PineconeAsyncio(api_key='YOUR_API_KEY')

            # Do async things
            index_list = await pc.list_indexes()

            # You're responsible for calling this yourself
            await pc.close()

        asyncio.run(main())
        ```

        Failing to do this may result in error messages appearing from the underlyling aiohttp library.

        ### Configuration with environment variables

        If you instantiate the Pinecone client with no arguments, it will attempt to read the API key from the environment variable `PINECONE_API_KEY`.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio() as pc:
                # Do async things
                index_list = await pc.list_indexes()

        asyncio.run(main())
        ```

        ### Configuration with keyword arguments

        If you prefer being more explicit in your code, you can also pass the API  as

        ### Configuration with environment variables

        If you instantiate the Pinecone client with no arguments, it will attempt to read the API key from the environment variable `PINECONE_API_KEY`.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio() as pc:
                # Do async things
                index_list = await pc.list_indexes()

        asyncio.run(main())
        ```

        ### Configuration with keyword arguments

        If you prefer being more explicit in your code, you can also pass the API  as



        ### Configuration with environment variables

        If you instantiate the Pinecone client with no arguments, it will attempt to read the API key from the environment variable `PINECONE_API_KEY`.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio() as pc:
                # Do async things
                index_list = await pc.list_indexes()

        asyncio.run(main())
        ```

        ### Configuration with keyword arguments

        If you prefer being more explicit in your code, you can also pass the API  as a keyword argument.

        ```python
        import os
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with Pinecone(api_key=os.environ.get("PINECONE_API_KEY")) as pc:
                # Do async things
                index_list = await pc.list_indexes()

        asyncio.run(main())
        ```

        ### Environment variables

        The Pinecone client supports the following environment variables:

        - `PINECONE_API_KEY`: The API key to use for authentication. If not passed via
        kwarg, the API key will be read from the environment variable `PINECONE_API_KEY`.

        ### Proxy configuration

        If your network setup requires you to interact with Pinecone via a proxy, you will need
        to pass additional configuration using optional keyword parameters. These optional parameters
        are used to configure an SSL context and passed to `aiohttp`, which is the underlying library
        currently used by the PineconeAsyncio client to make HTTP requests.

        Here is a basic example:

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio(
                api_key='YOUR_API_KEY',
                proxy_url='https://your-proxy.com'
            ) as pc:
                # Do async things
                index_list = await pc.list_indexes()

        asyncio.run(main())
        ```

        ### Using proxies with self-signed certificates

        By default the Pinecone Python client will perform SSL certificate verification
        using the CA bundle maintained by Mozilla in the [certifi](https://pypi.org/project/certifi/) package.
        If your proxy server is using a self-signed certificate, you will need to pass the path to the certificate
        in PEM format using the `ssl_ca_certs` parameter.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio(
                api_key='YOUR_API_KEY',
                proxy_url='https://your-proxy.com',
                ssl_ca_certs='path/to/cert-bundle.pem'
            ) as pc:
                # Do async things
                await pc.list_indexes()

        asyncio.run(main())
        ```

        ### Disabling SSL verification

        If you would like to disable SSL verification, you can pass the `ssl_verify`
        parameter with a value of `False`. We do not recommend going to production with SSL verification disabled
        but there are situations where this is useful such as testing with Pinecone Local running in a docker
        container.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio(
                api_key='YOUR_API_KEY',
                ssl_verify=False
            ) as pc:
                if not await pc.has_index('my_index'):
                    await pc.create_index(
                        name='my_index',
                        dimension=1536,
                        metric='cosine',
                        spec=ServerlessSpec(cloud='aws', region='us-west-2')
                    )

        asyncio.run(main())
        ```

        ### Passing additional headers

        If you need to pass additional headers with each request to the Pinecone API, you can do so using the
        `additional_headers` parameter. This is primarily for internal testing and end-users shouldn't need to
        do this unless specifically instructed to do so.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio(
                api_key='YOUR_API_KEY',
                host='https://api-staging.pinecone.io',
                additional_headers={'X-My-Header': 'my-value'}
            ) as pc:
                # Do async things
                await pc.list_indexes()

        asyncio.run(main())
        ```
        """

    pass

    @abstractmethod
    async def create_index(
        self,
        name: str,
        spec: Union[Dict, "ServerlessSpec", "PodSpec", "ByocSpec"],
        dimension: Optional[int],
        metric: Optional[Union["Metric", str]] = "cosine",
        timeout: Optional[int] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        vector_type: Optional[Union["VectorType", str]] = "dense",
        tags: Optional[Dict[str, str]] = None,
    ):
        """Creates a Pinecone index.

        :param name: The name of the index to create. Must be unique within your project and
            cannot be changed once created. Allowed characters are lowercase letters, numbers,
            and hyphens and the name may not begin or end with hyphens. Maximum length is 45 characters.
        :type name: str
        :param metric: Type of similarity metric used in the vector index when querying, one of `{"cosine", "dotproduct", "euclidean"}`.
        :type metric: str, optional
        :param spec: A dictionary containing configurations describing how the index should be deployed. For serverless indexes,
            specify region and cloud. For pod indexes, specify replicas, shards, pods, pod_type, metadata_config, and source_collection.
            Alternatively, use the `ServerlessSpec` or `PodSpec` objects to specify these configurations.
        :type spec: Dict
        :param dimension: If you are creating an index with `vector_type="dense"` (which is the default), you need to specify `dimension` to indicate the size of your vectors.
            This should match the dimension of the embeddings you will be inserting. For example, if you are using
            OpenAI's CLIP model, you should use `dimension=1536`. Dimension is a required field when
            creating an index with `vector_type="dense"` and should not be passed when `vector_type="sparse"`.
        :type dimension: int
        :type timeout: int, optional
        :param timeout: Specify the number of seconds to wait until index gets ready. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait.
        :param deletion_protection: If enabled, the index cannot be deleted. If disabled, the index can be deleted.
        :type deletion_protection: Optional[Literal["enabled", "disabled"]]
        :param vector_type: The type of vectors to be stored in the index. One of `{"dense", "sparse"}`.
        :type vector_type: str, optional
        :param tags: Tags are key-value pairs you can attach to indexes to better understand, organize, and identify your resources. Some example use cases include tagging indexes with the name of the model that generated the embeddings, the date the index was created, or the purpose of the index.
        :type tags: Optional[Dict[str, str]]
        :return: A `IndexModel` instance containing a description of the index that was created.

        ### Creating a serverless index

        ```python
        import os
        import asyncio

        from pinecone import (
            PineconeAsyncio,
            ServerlessSpec,
            CloudProvider,
            AwsRegion,
            Metric,
            DeletionProtection,
            VectorType
        )

        async def main():
            async with PineconeAsyncio(api_key=os.environ.get("PINECONE_API_KEY")) as pc:
                await pc.create_index(
                    name="my_index",
                    dimension=1536,
                    metric=Metric.COSINE,
                    spec=ServerlessSpec(
                        cloud=CloudProvider.AWS,
                        region=AwsRegion.US_WEST_2
                    ),
                    deletion_protection=DeletionProtection.DISABLED,
                    vector_type=VectorType.DENSE,
                    tags={
                        "model": "clip",
                        "app": "image-search",
                        "env": "testing"
                    }
                )

        asyncio.run(main())
        ```

        ### Creating a pod index

        ```python
        import os
        import asyncio

        from pinecone import (
            Pinecone,
            PodSpec,
            PodIndexEnvironment,
            PodType,
            Metric,
            DeletionProtection,
            VectorType
        )

        async def main():
            async with Pinecone(api_key=os.environ.get("PINECONE_API_KEY")) as pc:
                await pc.create_index(
                    name="my_index",
                    dimension=1536,
                    metric=Metric.COSINE,
                    spec=PodSpec(
                        environment=PodIndexEnvironment.US_EAST4_GCP,
                        pod_type=PodType.P1_X1
                    ),
                    deletion_protection=DeletionProtection.DISABLED,
                    tags={
                        "model": "clip",
                        "app": "image-search",
                        "env": "testing"
                    }
                )

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def create_index_for_model(
        self,
        name: str,
        cloud: Union["CloudProvider", str],
        region: Union["AwsRegion", "GcpRegion", "AzureRegion", str],
        embed: Union["IndexEmbed", "CreateIndexForModelEmbedTypedDict"],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        timeout: Optional[int] = None,
    ) -> "IndexModel":
        """
        :param name: The name of the index to create. Must be unique within your project and
            cannot be changed once created. Allowed characters are lowercase letters, numbers,
            and hyphens and the name may not begin or end with hyphens. Maximum length is 45 characters.
        :type name: str
        :param cloud: The cloud provider to use for the index. One of `{"aws", "gcp", "azure"}`.
        :type cloud: str
        :param region: The region to use for the index. Enum objects `AwsRegion`, `GcpRegion`, and `AzureRegion` are also available to help you quickly set these parameters, but may not be up to date as new regions become available.
        :type region: str
        :param embed: The embedding configuration for the index. This param accepts a dictionary or an instance of the `IndexEmbed` object.
        :type embed: Union[Dict, IndexEmbed]
        :param tags: Tags are key-value pairs you can attach to indexes to better understand, organize, and identify your resources. Some example use cases include tagging indexes with the name of the model that generated the embeddings, the date the index was created, or the purpose of the index.
        :type tags: Optional[Dict[str, str]]
        :param deletion_protection: If enabled, the index cannot be deleted. If disabled, the index can be deleted. This setting can be changed with `configure_index`.
        :type deletion_protection: Optional[Literal["enabled", "disabled"]]
        :type timeout: Optional[int]
        :param timeout: Specify the number of seconds to wait until index is ready to receive data. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait.
        :return: A description of the index that was created.
        :rtype: IndexModel

        This method is used to create a Serverless index that is configured for use with Pinecone's integrated inference models.

        The resulting index can be described, listed, configured, and deleted like any other Pinecone index with the `describe_index`, `list_indexes`, `configure_index`, and `delete_index` methods.

        After the model is created, you can upsert records into the index with the `upsert_records` method, and search your records with the `search` method.

        ```python
        import asyncio

        from pinecone import (
            PineconeAsyncio,
            IndexEmbed,
            CloudProvider,
            AwsRegion,
            EmbedModel,
            Metric,
        )

        async def main():
            async with PineconeAsyncio() as pc:
                if not await pc.has_index("book-search"):
                    desc = await pc.create_index_for_model(
                        name="book-search",
                        cloud=CloudProvider.AWS,
                        region=AwsRegion.US_EAST_1,
                        embed=IndexEmbed(
                            model=EmbedModel.Multilingual_E5_Large,
                            metric=Metric.COSINE,
                            field_map={
                                "text": "description",
                            },
                        )
                    )

        asyncio.run(main())
        ```

        To see the available cloud regions, see this [Pinecone documentation](https://docs.pinecone.io/troubleshooting/available-cloud-regions) page.

        See the [Model Gallery](https://docs.pinecone.io/models/overview) to learn about available models.
        """
        pass

    @abstractmethod
    def create_index_from_backup(
        self,
        *,
        name: str,
        backup_id: str,
        deletion_protection: Optional[Union["DeletionProtection", str]] = "disabled",
        tags: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> "IndexModel":
        """
        Create an index from a backup.

        Call `list_backups` to get a list of backups for your project.

        :param name: The name of the index to create.
        :type name: str
        :param backup_id: The ID of the backup to restore.
        :type backup_id: str
        :param deletion_protection: If enabled, the index cannot be deleted. If disabled, the index can be deleted. This setting can be changed with `configure_index`.
        :type deletion_protection: Optional[Literal["enabled", "disabled"]]
        :param tags: Tags are key-value pairs you can attach to indexes to better understand, organize, and identify your resources. Some example use cases include tagging indexes with the name of the model that generated the embeddings, the date the index was created, or the purpose of the index.
        :type tags: Optional[Dict[str, str]]
        :param timeout: Specify the number of seconds to wait until index is ready to receive data. If None, wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait.
        :return: A description of the index that was created.
        :rtype: IndexModel
        """
        pass

    @abstractmethod
    async def delete_index(self, name: str, timeout: Optional[int] = None):
        """
        :param name: the name of the index.
        :type name: str
        :param timeout: Number of seconds to poll status checking whether the index has been deleted. If None,
            wait indefinitely; if >=0, time out after this many seconds;
            if -1, return immediately and do not wait.
        :type timeout: int, optional

        Deletes a Pinecone index.

        Deleting an index is an irreversible operation. All data in the index will be lost.
        When you use this command, a request is sent to the Pinecone control plane to delete
        the index, but the termination is not synchronous because resources take a few moments to
        be released.

        By default the `delete_index` method will block until polling of the `describe_index` method
        shows that the delete operation has completed. If you prefer to return immediately and not
        wait for the index to be deleted, you can pass `timeout=-1` to the method.

        After the delete request is submitted, polling `describe_index` will show that the index
        transitions into a `Terminating` state before eventually resulting in a 404 after it has been removed.

        This operation can fail if the index is configured with `deletion_protection="enabled"`.
        In this case, you will need to call `configure_index` to disable deletion protection before
        you can delete the index.

        ```python
        import asyncio

        from pinecone import PineconeAsyncio

        async def main():
            pc = PineconeAsyncio()

            index_name = "my_index"
            desc = await pc.describe_index(name=index_name)

            if desc.deletion_protection == "enabled":
                # If for some reason deletion protection is enabled, you will need to disable it first
                # before you can delete the index. But use caution as this operation is not reversible
                # and if somebody enabled deletion protection, they probably had a good reason.
                await pc.configure_index(name=index_name, deletion_protection="disabled")

            await pc.delete_index(name=index_name)
            await pc.close()

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def list_indexes(self) -> "IndexList":
        """
        :return: Returns an `IndexList` object, which is iterable and contains a
            list of `IndexModel` objects. The `IndexList` also has a convenience method `names()`
            which returns a list of index names for situations where you just want to iterate over
            all index names.

        Lists all indexes in your project.

        The results include a description of all indexes in your project, including the
        index name, dimension, metric, status, and spec.

        If you simply want to check whether an index exists, see the `has_index()` convenience method.

        You can use the `list_indexes()` method to iterate over descriptions of every index in your project.

        ```python
        import asyncio

        from pinecone import PineconeAsyncio

        async def main():
            pc = PineconeAsyncio()

            available_indexes = await pc.list_indexes()
            for index in available_indexes:
                print(index.name)
                print(index.dimension)
                print(index.metric)
                print(index.status)
                print(index.host)
                print(index.spec)

            await pc.close()

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def describe_index(self, name: str) -> "IndexModel":
        """
        :param name: the name of the index to describe.
        :return: Returns an `IndexModel` object
        which gives access to properties such as the
        index name, dimension, metric, host url, status,
        and spec.

        Describes a Pinecone index.

        ### Getting your index host url

        In a real production situation, you probably want to
        store the host url in an environment variable so you
        don't have to call describe_index and re-fetch it
        every time you want to use the index. But this example
        shows how to get the value from the API using describe_index.

        ```python
        import asyncio
        from pinecone import Pinecone, PineconeAsyncio, Index

        async def main():
            pc = PineconeAsyncio()

            index_name="my_index"
            description = await pc.describe_index(name=index_name)
            print(description)
            # {
            #     "name": "my_index",
            #     "metric": "cosine",
            #     "host": "my_index-dojoi3u.svc.aped-4627-b74a.pinecone.io",
            #     "spec": {
            #         "serverless": {
            #             "cloud": "aws",
            #             "region": "us-east-1"
            #         }
            #     },
            #     "status": {
            #         "ready": true,
            #         "state": "Ready"
            #     },
            #     "vector_type": "dense",
            #     "dimension": 1024,
            #     "deletion_protection": "enabled",
            #     "tags": {
            #         "environment": "production"
            #     }
            # }

            print(f"Your index is hosted at {description.host}")
            await pc.close()

            async with Pinecone().IndexAsyncio(host=description.host) as idx:
                await idx.upsert(vectors=[...])

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def has_index(self, name: str) -> bool:
        """
        :param name: The name of the index to check for existence.
        :return: Returns `True` if the index exists, `False` otherwise.

        Checks if a Pinecone index exists.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio, ServerlessSpec

        async def main():
            async with PineconeAsyncio() as pc:
                index_name = "my_index"
                if not await pc.has_index(index_name):
                    print("Index does not exist, creating...")
                    pc.create_index(
                        name=index_name,
                        dimension=768,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-west-2")
                    )

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def configure_index(
        self,
        name: str,
        replicas: Optional[int] = None,
        pod_type: Optional[Union["PodType", str]] = None,
        deletion_protection: Optional[Union["DeletionProtection", str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        :param: name: the name of the Index
        :param: replicas: the desired number of replicas, lowest value is 0.
        :param: pod_type: the new pod_type for the index. To learn more about the
            available pod types, please see [Understanding Indexes](https://docs.pinecone.io/docs/indexes)
        :param: deletion_protection: If set to 'enabled', the index cannot be deleted. If 'disabled', the index can be deleted.
        :param: tags: A dictionary of tags to apply to the index. Tags are key-value pairs that can be used to organize and manage indexes. To remove a tag, set the value to "". Tags passed to configure_index will be merged with existing tags and any with the value empty string will be removed.

        This method is used to modify an index's configuration. It can be used to:

        - Scale a pod-based index horizontally using `replicas`
        - Scale a pod-based index vertically using `pod_type`
        - Enable or disable deletion protection using `deletion_protection`
        - Add, change, or remove tags using `tags`

        ## Scaling pod-based indexes

        To scale your pod-based index, you pass a `replicas` and/or `pod_type` param to the `configure_index` method. `pod_type` may be a string or a value from the `PodType` enum.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio, PodType

        async def main():
            async with PineconeAsyncio() as pc:
                await pc.configure_index(
                    name="my_index",
                    replicas=2,
                    pod_type=PodType.P1_X2
                )

        asyncio.run(main())
        ```

        After providing these new configurations, you must call `describe_index` to see the status of the index as the changes are applied.

        ## Enabling or disabling deletion protection

        To enable or disable deletion protection, pass the `deletion_protection` parameter to the `configure_index` method. When deletion protection
        is enabled, the index cannot be deleted with the `delete_index` method.

        ```python
        import asyncio
        from pinecone import PineconeAsyncio, DeletionProtection

        async def main():
            async with PineconeAsyncio() as pc:
                # Enable deletion protection
                await pc.configure_index(
                    name="my_index",
                    deletion_protection=DeletionProtection.ENABLED
                )

                # Call describe_index to see the change was applied.
                desc = await pc.describe_index("my_index")
                assert desc.deletion_protection == "enabled"

                # Disable deletion protection
                await pc.configure_index(
                    name="my_index",
                    deletion_protection=DeletionProtection.DISABLED
                )

        asyncio.run(main())
        ```

        ## Adding, changing, or removing tags

        To add, change, or remove tags, pass the `tags` parameter to the `configure_index` method. When tags are passed using `configure_index`,
        they are merged with any existing tags already on the index. To remove a tag, set the value of the key to an empty string.

        ```python
        import asyncio

        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio() as pc:
                # Add a tag
                await pc.configure_index(name="my_index", tags={"environment": "staging"})

                # Change a tag
                await pc.configure_index(name="my_index", tags={"environment": "production"})

                # Remove a tag
                await pc.configure_index(name="my_index", tags={"environment": ""})

                # Call describe_index to view the tags are changed
                await pc.describe_index("my_index")
                print(desc.tags)

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def create_backup(
        self, *, index_name: str, backup_name: str, description: str = ""
    ) -> "BackupModel":
        """Create a backup of an index.

        Args:
            index_name (str): The name of the index to backup.
            backup_name (str): The name to give the backup.
            description (str): Optional description of the backup.
        """
        pass

    @abstractmethod
    async def list_backups(
        self,
        *,
        index_name: Optional[str] = None,
        limit: Optional[int] = 10,
        pagination_token: Optional[str] = None,
    ) -> "BackupList":
        """List backups.

        If index_name is provided, the backups will be filtered by index. If no index_name is provided, all backups in the projectwill be returned.

        Args:
            index_name (str): The name of the index to list backups for.
            limit (int): The maximum number of backups to return.
            pagination_token (str): The pagination token to use for pagination.
        """
        pass

    @abstractmethod
    async def describe_backup(self, *, backup_id: str) -> "BackupModel":
        """Describe a backup.

        Args:
            backup_id (str): The ID of the backup to describe.
        """
        pass

    @abstractmethod
    async def delete_backup(self, *, backup_id: str) -> None:
        """Delete a backup.

        Args:
            backup_id (str): The ID of the backup to delete.
        """
        pass

    @abstractmethod
    async def list_restore_jobs(
        self, *, limit: Optional[int] = 10, pagination_token: Optional[str] = None
    ) -> "RestoreJobList":
        """List restore jobs.

        Args:
            limit (int): The maximum number of restore jobs to return.
            pagination_token (str): The pagination token to use for pagination.
        """
        pass

    @abstractmethod
    async def describe_restore_job(self, *, job_id: str) -> "RestoreJobModel":
        """Describe a restore job.

        Args:
            job_id (str): The ID of the restore job to describe.
        """
        pass

    @abstractmethod
    async def create_collection(self, name: str, source: str):
        """Create a collection from a pod-based index

        :param name: Name of the collection
        :param source: Name of the source index
        """
        pass

    @abstractmethod
    async def list_collections(self) -> "CollectionList":
        """List all collections

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            pc = PineconeAsyncio()

            collections = await pc.list_collections()
            for collection in collections:
                print(collection.name)
                print(collection.source)

            # You can also iterate specifically over
            # a list of collection names by calling
            # the .names() helper.
            collection_name = "my_collection"
            collections = await pc.list_collections()
            if collection_name in collections.names():
                print('Collection exists')

            await pc.close()

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def delete_collection(self, name: str):
        """Describes a collection.
        :param: The name of the collection
        :return: Description of the collection

        ```python
        import asyncio
        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio() as pc:

                description = await pc.describe_collection("my_collection")
                print(description.name)
                print(description.source)
                print(description.status)
                print(description.size)

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def describe_collection(self, name: str):
        """Describes a collection.
        :param: The name of the collection
        :return: Description of the collection

        ```python
        import asyncio
        from pinecone import PineconeAsyncio


        async def main():
            async with PineconeAsyncio() as pc:
                description = await pc.describe_collection("my_collection")
                print(description.name)
                print(description.source)
                print(description.status)
                print(description.size)

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    def IndexAsyncio(self, host, **kwargs):
        """
        Build an asyncio-compatible client for index data operations.

        :param host: The host url of the index.

        ```python
        import os
        import asyncio

        from pinecone import PineconeAsyncio

        api_key = os.environ.get("PINECONE_API_KEY")
        index_host = os.environ.get("PINECONE_INDEX_HOST")

        async def main():
            async with Pinecone(api_key=api_key) as pc:
                async with pc.Index(host=index_host) as idx:
                    # Now you're ready to perform data operations
                    await index.query(vector=[...], top_k=10)

        asyncio.run(main())
        ```

        To find your host url, you can use the `describe_index`. Or, alternatively, the
        host is displayed in the Pinecone web console.

        ```python
        import os
        import asyncio

        from pinecone import PineconeAsyncio

        async def main():
            async with PineconeAsyncio(
                api_key=os.environ.get("PINECONE_API_KEY")
            ) as pc:
                host = await pc.describe_index('index-name').host

        asyncio.run(main())
        ```

        ## Alternative setup

        Like instances of the `PineconeAsyncio` class, instances of `IndexAsyncio` have async context that
        needs to be cleaned up when you are done with it in order to avoid error messages about unclosed session from
        aiohttp. Nesting these in code is a bit cumbersome, so if you are only planning to do data operations you
        may prefer to setup the `IndexAsyncio` object via the `Pinecone` class which will avoid creating an outer async context.

        ```python
        import os
        import asyncio
        from pinecone import Pinecone

        api_key = os.environ.get("PINECONE_API_KEY")

        async def main():
            pc = Pinecone(api_key=api_key) # sync client, so no async context to worry about

            async with pc.AsyncioIndex(host='your_index_host') as idx:
                # Now you're ready to perform data operations
                await idx.query(vector=[...], top_k=10)

        ```
        """
        pass
