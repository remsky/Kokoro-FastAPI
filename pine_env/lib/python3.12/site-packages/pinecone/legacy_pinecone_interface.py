from abc import ABC, abstractmethod

from typing import Optional, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
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


class LegacyPineconeDBControlInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        proxy_url: Optional[str] = None,
        proxy_headers: Optional[Dict[str, str]] = None,
        ssl_ca_certs: Optional[str] = None,
        ssl_verify: Optional[bool] = None,
        additional_headers: Optional[Dict[str, str]] = {},
        pool_threads: Optional[int] = 1,
        **kwargs,
    ):
        """
        The `Pinecone` class is the main entry point for interacting with Pinecone via this Python SDK.
        Instances of the `Pinecone` class are used to create, delete, and manage your indexes and collections. The class also holds inference functionality (embed, rerank) under the `inference` namespace.
        Methods which create or modify index and collection resources result in network calls to https://api.pinecone.io.

        When you are ready to perform data operations on an index, you will need to instantiate an index client. Though the functionality of the index client is defined in a different
        class, it is instantiated through the `Index()` method in order for configurations to be shared between the two objects.

        :param api_key: The API key to use for authentication. If not passed via kwarg, the API key will be read from the environment variable `PINECONE_API_KEY`.
        :type api_key: str, optional
        :param host: The control plane host. If unspecified, the host `api.pinecone.io` will be used.
        :type host: str, optional
        :param proxy_url: The URL of the proxy to use for the connection.
        :type proxy_url: str, optional
        :param proxy_headers: Additional headers to pass to the proxy. Use this if your proxy setup requires authentication.
        :type proxy_headers: Dict[str, str], optional
        :param ssl_ca_certs: The path to the SSL CA certificate bundle to use for the connection. This path should point to a file in PEM format. When not passed, the SDK will use the certificate bundle returned from `certifi.where()`.
        :type ssl_ca_certs: str, optional
        :param ssl_verify: SSL verification is performed by default, but can be disabled using the boolean flag when testing with Pinecone Local or troubleshooting a proxy setup. You should never run with SSL verification disabled in production.
        :type ssl_verify: bool, optional
        :param additional_headers: Additional headers to pass to the API. This is mainly to support internal testing at Pinecone. End users should not need to use this unless following specific instructions to do so.
        :type additional_headers: Dict[str, str], optional
        :param pool_threads: The number of threads to use for the ThreadPool when using methods that support the `async_req` keyword argument. The default number of threads is 5 * the number of CPUs in your execution environment.
        :type pool_threads: int, optional

        ### Configuration with environment variables

        If you instantiate the Pinecone client with no arguments, it will attempt to read the API key from the environment variable `PINECONE_API_KEY`.

        ```python
        from pinecone import Pinecone

        pc = Pinecone()
        ```

        ### Configuration with keyword arguments

        If you prefer being more explicit in your code, you can also pass the API key as a keyword argument. This is also where you will pass additional configuration options such as proxy settings if you wish to use those.

        ```python
        import os
        from pinecone import Pinecone

        pc = Pinecone(
            api_key=os.environ.get("PINECONE_API_KEY"),
            host="https://api-staging.pinecone.io"
        )
        ```

        ### Environment variables

        The Pinecone client supports the following environment variables:

        - `PINECONE_API_KEY`: The API key to use for authentication. If not passed via
        kwarg, the API key will be read from the environment variable `PINECONE_API_KEY`.

        - `PINECONE_DEBUG_CURL`: When troubleshooting it can be very useful to run curl
        commands against the control plane API to see exactly what data is being sent
        and received without all the abstractions and transformations applied by the Python
        SDK. If you set this environment variable to `true`, the Pinecone client will use
        request parameters to print out an equivalent curl command that you can run yourself
        or share with Pinecone support. **Be very careful with this option, as it will print out
        your API key** which forms part of a required authentication header. The main use of
        is to help evaluate whether a problem you are experiencing is due to the API's behavior
        or the behavior of the SDK itself.

        ### Proxy configuration

        If your network setup requires you to interact with Pinecone via a proxy, you will need
        to pass additional configuration using optional keyword parameters. These optional parameters
        are forwarded to `urllib3`, which is the underlying library currently used by the Pinecone client to
        make HTTP requests. You may find it helpful to refer to the
        [urllib3 documentation on working with proxies](https://urllib3.readthedocs.io/en/stable/advanced-usage.html#http-and-https-proxies)
        while troubleshooting these settings.

        Here is a basic example:

        ```python
        from pinecone import Pinecone

        pc = Pinecone(
            api_key='YOUR_API_KEY',
            proxy_url='https://your-proxy.com'
        )

        pc.list_indexes()
        ```

        If your proxy requires authentication, you can pass those values in a header dictionary using the `proxy_headers` parameter.

        ```python
        from pinecone import Pinecone
        import urllib3 import make_headers

        pc = Pinecone(
            api_key='YOUR_API_KEY',
            proxy_url='https://your-proxy.com',
            proxy_headers=make_headers(proxy_basic_auth='username:password')
        )

        pc.list_indexes()
        ```

        ### Using proxies with self-signed certificates

        By default the Pinecone Python client will perform SSL certificate verification
        using the CA bundle maintained by Mozilla in the [certifi](https://pypi.org/project/certifi/) package.
        If your proxy server is using a self-signed certificate, you will need to pass the path to the certificate
        in PEM format using the `ssl_ca_certs` parameter.

        ```python
        from pinecone import Pinecone
        import urllib3 import make_headers

        pc = Pinecone(
            api_key='YOUR_API_KEY',
            proxy_url='https://your-proxy.com',
            proxy_headers=make_headers(proxy_basic_auth='username:password'),
            ssl_ca_certs='path/to/cert-bundle.pem'
        )

        pc.list_indexes()
        ```

        ### Disabling SSL verification

        If you would like to disable SSL verification, you can pass the `ssl_verify`
        parameter with a value of `False`. We do not recommend going to production with SSL verification disabled.

        ```python
        from pinecone import Pinecone
        import urllib3 import make_headers

        pc = Pinecone(
            api_key='YOUR_API_KEY',
            proxy_url='https://your-proxy.com',
            proxy_headers=make_headers(proxy_basic_auth='username:password'),
            ssl_ca_certs='path/to/cert-bundle.pem',
            ssl_verify=False
        )

        pc.list_indexes()

        ```
        """

    pass

    @abstractmethod
    def create_index(
        self,
        name: str,
        spec: Union[Dict, "ServerlessSpec", "PodSpec", "ByocSpec"],
        dimension: Optional[int],
        metric: Optional[Union["Metric", str]] = "Metric.COSINE",
        timeout: Optional[int] = None,
        deletion_protection: Optional[
            Union["DeletionProtection", str]
        ] = "DeletionProtection.DISABLED",
        vector_type: Optional[Union["VectorType", str]] = "VectorType.DENSE",
        tags: Optional[Dict[str, str]] = None,
    ) -> "IndexModel":
        """Creates a Pinecone index.

        :param name: The name of the index to create. Must be unique within your project and
            cannot be changed once created. Allowed characters are lowercase letters, numbers,
            and hyphens and the name may not begin or end with hyphens. Maximum length is 45 characters.
        :type name: str
        :param metric: Type of similarity metric used in the vector index when querying, one of `{"cosine", "dotproduct", "euclidean"}`.
        :type metric: str, optional
        :param spec: A dictionary containing configurations describing how the index should be deployed. For serverless indexes,
            specify region and cloud. For pod indexes, specify replicas, shards, pods, pod_type, metadata_config, and source_collection.
            Alternatively, use the `ServerlessSpec`, `PodSpec`, or `ByocSpec` objects to specify these configurations.
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
        from pinecone import (
            Pinecone,
            ServerlessSpec,
            CloudProvider,
            AwsRegion,
            Metric,
            DeletionProtection,
            VectorType
        )

        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        pc.create_index(
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
        ```

        ### Creating a pod index

        ```python
        import os
        from pinecone import (
            Pinecone,
            PodSpec,
            PodIndexEnvironment,
            PodType,
            Metric,
            DeletionProtection,
            VectorType
        )

        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        pc.create_index(
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
        ```
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
    def create_index_for_model(
        self,
        *,
        name: str,
        cloud: Union["CloudProvider", str],
        region: Union["AwsRegion", "GcpRegion", "AzureRegion", str],
        embed: Union["IndexEmbed", "CreateIndexForModelEmbedTypedDict"],
        tags: Optional[Dict[str, str]] = None,
        deletion_protection: Optional[
            Union["DeletionProtection", str]
        ] = "DeletionProtection.DISABLED",
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
        from pinecone import (
            Pinecone,
            IndexEmbed,
            CloudProvider,
            AwsRegion,
            EmbedModel,
            Metric,
        )

        pc = Pinecone()

        if not pc.has_index("book-search"):
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
        ```

        To see the available cloud regions, see this [Pinecone documentation](https://docs.pinecone.io/troubleshooting/available-cloud-regions) page.

        See the [Model Gallery](https://docs.pinecone.io/models/overview) to learn about available models.
        """
        pass

    @abstractmethod
    def delete_index(self, name: str, timeout: Optional[int] = None):
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
        from pinecone import Pinecone

        pc = Pinecone()

        index_name = "my_index"
        desc = pc.describe_index(name=index_name)

        if desc.deletion_protection == "enabled":
            # If for some reason deletion protection is enabled, you will need to disable it first
            # before you can delete the index. But use caution as this operation is not reversible
            # and if somebody enabled deletion protection, they probably had a good reason.
            pc.configure_index(name=index_name, deletion_protection="disabled")

        pc.delete_index(name=index_name)
        ```
        """
        pass

    @abstractmethod
    def list_indexes(self) -> "IndexList":
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
        from pinecone import Pinecone

        pc = Pinecone()

        for index in pc.list_indexes():
            print(index.name)
            print(index.dimension)
            print(index.metric)
            print(index.status)
            print(index.host)
            print(index.spec)
        ```
        """
        pass

    @abstractmethod
    def describe_index(self, name: str) -> "IndexModel":
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
        from pinecone import Pinecone, Index

        pc = Pinecone()

        index_name="my_index"
        description = pc.describe_index(name=index_name)
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

        index = pc.Index(host=description.host)
        index.upsert(vectors=[...])
        ```
        """
        pass

    @abstractmethod
    def has_index(self, name: str) -> bool:
        """
        :param name: The name of the index to check for existence.
        :return: Returns `True` if the index exists, `False` otherwise.

        Checks if a Pinecone index exists.

        ```python
        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone()

        index_name = "my_index"
        if not pc.has_index(index_name):
            print("Index does not exist, creating...")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
        ```
        """
        pass

    @abstractmethod
    def configure_index(
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
        from pinecone import Pinecone, PodType

        pc = Pinecone()
        pc.configure_index(
            name="my_index",
            replicas=2,
            pod_type=PodType.P1_X2
        )
        ```

        After providing these new configurations, you must call `describe_index` to see the status of the index as the changes are applied.

        ## Enabling or disabling deletion protection

        To enable or disable deletion protection, pass the `deletion_protection` parameter to the `configure_index` method. When deletion protection
        is enabled, the index cannot be deleted with the `delete_index` method.

        ```python
        from pinecone import Pinecone, DeletionProtection

        pc = Pinecone()

        # Enable deletion protection
        pc.configure_index(
            name="my_index",
            deletion_protection=DeletionProtection.ENABLED
        )

        # Call describe_index to see the change was applied.
        assert pc.describe_index("my_index").deletion_protection == "enabled"

        # Disable deletion protection
        pc.configure_index(
            name="my_index",
            deletion_protection=DeletionProtection.DISABLED
        )
        ```

        ## Adding, changing, or removing tags

        To add, change, or remove tags, pass the `tags` parameter to the `configure_index` method. When tags are passed using `configure_index`,
        they are merged with any existing tags already on the index. To remove a tag, set the value of the key to an empty string.

        ```python
        from pinecone import Pinecone

        pc = Pinecone()

        # Add a tag
        pc.configure_index(name="my_index", tags={"environment": "staging"})

        # Change a tag
        pc.configure_index(name="my_index", tags={"environment": "production"})

        # Remove a tag
        pc.configure_index(name="my_index", tags={"environment": ""})

        # Call describe_index to view the tags are changed
        print(pc.describe_index("my_index").tags)
        ```
        """
        pass

    @abstractmethod
    def create_collection(self, name: str, source: str) -> None:
        """Create a collection from a pod-based index

        :param name: Name of the collection
        :param source: Name of the source index
        """
        pass

    @abstractmethod
    def list_collections(self) -> "CollectionList":
        """List all collections

        ```python
        from pinecone import Pinecone

        pc = Pinecone()

        for collection in pc.list_collections():
            print(collection.name)
            print(collection.source)

        # You can also iterate specifically over the collection
        # names with the .names() helper.
        collection_name="my_collection"
        for collection_name in pc.list_collections().names():
            print(collection_name)
        ```
        """
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """
        :param name: The name of the collection to delete.

        Deletes a collection.

        Deleting a collection is an irreversible operation. All data
        in the collection will be lost.

        This method tells Pinecone you would like to delete a collection,
        but it takes a few moments to complete the operation. Use the
        `describe_collection()` method to confirm that the collection
        has been deleted.

        ```python
        from pinecone import Pinecone

        pc = Pinecone()

        pc.delete_collection(name="my_collection")
        ```
        """
        pass

    @abstractmethod
    def describe_collection(self, name: str):
        """Describes a collection.
        :param: The name of the collection
        :return: Description of the collection

        ```python
        from pinecone import Pinecone

        pc = Pinecone()

        description = pc.describe_collection("my_collection")
        print(description.name)
        print(description.source)
        print(description.status)
        print(description.size)
        ```
        """
        pass

    @abstractmethod
    def create_backup(
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
    def list_backups(
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
    def describe_backup(self, *, backup_id: str) -> "BackupModel":
        """Describe a backup.

        Args:
            backup_id (str): The ID of the backup to describe.
        """
        pass

    @abstractmethod
    def delete_backup(self, *, backup_id: str) -> None:
        """Delete a backup.

        Args:
            backup_id (str): The ID of the backup to delete.
        """
        pass

    @abstractmethod
    def list_restore_jobs(
        self, *, limit: Optional[int] = 10, pagination_token: Optional[str] = None
    ) -> "RestoreJobList":
        """List restore jobs.

        Args:
            limit (int): The maximum number of restore jobs to return.
            pagination_token (str): The pagination token to use for pagination.
        """
        pass

    @abstractmethod
    def describe_restore_job(self, *, job_id: str) -> "RestoreJobModel":
        """Describe a restore job.

        Args:
            job_id (str): The ID of the restore job to describe.
        """
        pass

    @abstractmethod
    def Index(self, name: str = "", host: str = "", **kwargs):
        """
        :param name: The name of the index to target. If you specify the name of the index, the client will
            fetch the host url from the Pinecone control plane.
        :param host: The host url of the index to target. If you specify the host url, the client will use
            the host url directly without making any additional calls to the control plane.
        :param pool_threads: The number of threads to use when making parallel requests by calling index methods with optional kwarg async_req=True, or using methods that make use of thread-based parallelism automatically such as query_namespaces().
        :param connection_pool_maxsize: The maximum number of connections to keep in the connection pool.
        :return: An instance of the `Index` class.

        Target an index for data operations.

        ### Target an index by host url

        In production situations, you want to uspert or query your data as quickly
        as possible. If you know in advance the host url of your index, you can
        eliminate a round trip to the Pinecone control plane by specifying the
        host of the index. If instead you pass the name of the index, the client
        will need to make an additional call to api.pinecone.io to get the host url
        before any data operations can take place.

        ```python
        import os
        from pinecone import Pinecone

        api_key = os.environ.get("PINECONE_API_KEY")
        index_host = os.environ.get("PINECONE_INDEX_HOST")

        pc = Pinecone(api_key=api_key)
        index = pc.Index(host=index_host)

        # Now you're ready to perform data operations
        index.query(vector=[...], top_k=10)
        ```

        To find your host url, you can use the describe_index method to call api.pinecone.io.
        The host url is returned in the response. Or, alternatively, the
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
        recommended for production usage because it introduces an unnecessary runtime
        dependency on api.pinecone.io.

        ```python
        import os
        from pinecone import Pinecone, ServerlessSpec

        api_key = os.environ.get("PINECONE_API_KEY")

        pc = Pinecone(api_key=api_key)
        pc.create_index(
            name='my_index',
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )
        index = pc.Index('my_index')

        # Now you're ready to perform data operations
        index.query(vector=[...], top_k=10)
        ```
        """
        pass

    def IndexAsyncio(self, host: str, **kwargs):
        """Build an asyncio-compatible Index object.

        :param host: The host url of the index to target. You can find this url in the Pinecone
            web console or by calling describe_index method of `Pinecone` or `PineconeAsyncio`.

        :return: An instance of the `IndexAsyncio` class.
        """
        pass
