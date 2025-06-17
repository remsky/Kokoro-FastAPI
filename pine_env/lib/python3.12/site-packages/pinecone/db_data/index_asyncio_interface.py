from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict, Any

from pinecone.core.openapi.db_data.models import (
    FetchResponse,
    QueryResponse,
    IndexDescription as DescribeIndexStatsResponse,
    UpsertResponse,
    Vector,
    ListResponse,
    SparseValues,
    SearchRecordsResponse,
)
from .query_results_aggregator import QueryNamespacesResults
from .types import (
    VectorTypedDict,
    SparseVectorTypedDict,
    VectorMetadataTypedDict,
    FilterTypedDict,
    VectorTuple,
    VectorTupleWithMetadata,
    SearchQueryTypedDict,
    SearchRerankTypedDict,
)
from .dataclasses import SearchQuery, SearchRerank


class IndexAsyncioInterface(ABC):
    @abstractmethod
    async def upsert(
        self,
        vectors: Union[
            List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]
        ],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> UpsertResponse:
        """
        Args:
            vectors (Union[List[Vector], List[VectorTuple], List[VectorTupleWithMetadata], List[VectorTypedDict]]): A list of vectors to upsert.
            namespace (str): The namespace to write to. If not specified, the default namespace is used. [optional]
            batch_size (int): The number of vectors to upsert in each batch.
                               If not specified, all vectors will be upserted in a single batch. [optional]
            show_progress (bool): Whether to show a progress bar using tqdm.
                                  Applied only if batch_size is provided. Default is True.

        Returns:
            `UpsertResponse`, includes the number of vectors upserted.


        The upsert operation writes vectors into a namespace.
        If a new value is upserted for an existing vector id, it will overwrite the previous value.

        To upsert in parallel follow: https://docs.pinecone.io/docs/insert-data#sending-upserts-in-parallel

        ## Upserting dense vectors

        **Note:** the dimension of each dense vector must match the dimension of the index.

        A vector can be represented in a variety of ways.

        ```python
        import asyncio
        from pinecone import Pinecone, Vector

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                # A Vector object
                await idx.upsert(
                    namespace = 'my-namespace',
                    vectors = [
                        Vector(id='id1', values=[0.1, 0.2, 0.3, 0.4], metadata={'metadata_key': 'metadata_value'}),
                    ]
                )

                # A vector tuple
                await idx.upsert(
                    namespace = 'my-namespace',
                    vectors = [
                        ('id1', [0.1, 0.2, 0.3, 0.4]),
                    ]
                )

                # A vector tuple with metadata
                await idx.upsert(
                    namespace = 'my-namespace',
                    vectors = [
                        ('id1', [0.1, 0.2, 0.3, 0.4], {'metadata_key': 'metadata_value'}),
                    ]
                )

                # A vector dictionary
                await idx.upsert(
                    namespace = 'my-namespace',
                    vectors = [
                        {"id": 1, "values": [0.1, 0.2, 0.3, 0.4], "metadata": {"metadata_key": "metadata_value"}},
                    ]

        asyncio.run(main())
        ```

        ## Upserting sparse vectors

        ```python
        import asyncio
        from pinecone import Pinecone, Vector, SparseValues

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                # A Vector object
                await idx.upsert(
                    namespace = 'my-namespace',
                    vectors = [
                        Vector(id='id1', sparse_values=SparseValues(indices=[1, 2], values=[0.2, 0.4])),
                    ]
                )

                # A dictionary
                await idx.upsert(
                    namespace = 'my-namespace',
                    vectors = [
                        {"id": 1, "sparse_values": {"indices": [1, 2], "values": [0.2, 0.4]}},
                    ]
                )

        asyncio.run(main())
        ```

        ## Batch upsert

        If you have a large number of vectors, you can upsert them in batches.

        ```python
        import asyncio
        from pinecone import Pinecone, Vector, SparseValues

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:

            await idx.upsert(
                namespace = 'my-namespace',
                vectors = [
                    {'id': 'id1', 'values': [0.1, 0.2, 0.3, 0.4]},
                    {'id': 'id2', 'values': [0.2, 0.3, 0.4, 0.5]},
                    {'id': 'id3', 'values': [0.3, 0.4, 0.5, 0.6]},
                    {'id': 'id4', 'values': [0.4, 0.5, 0.6, 0.7]},
                    {'id': 'id5', 'values': [0.5, 0.6, 0.7, 0.8]},
                    # More vectors here
                ],
                batch_size = 50
            )

        asyncio.run(main())
        ```

        ## Visual progress bar with tqdm

        To see a progress bar when upserting in batches, you will need to separately install the `tqdm` package.
        If `tqdm` is present, the client will detect and use it to display progress when `show_progress=True`.
        """
        pass

    @abstractmethod
    async def upsert_from_dataframe(
        self, df, namespace: Optional[str] = None, batch_size: int = 500, show_progress: bool = True
    ):
        """This method has not been implemented yet for the IndexAsyncio class."""
        pass

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            ids (List[str]): Vector ids to delete [optional]
            delete_all (bool): This indicates that all vectors in the index namespace should be deleted.. [optional]
                                Default is False.
            namespace (str): The namespace to delete vectors from [optional]
                            If not specified, the default namespace is used.
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
                    If specified, the metadata filter here will be used to select the vectors to delete.
                    This is mutually exclusive with specifying ids to delete in the ids param or using delete_all=True.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]


        The Delete operation deletes vectors from the index, from a single namespace.

        No error is raised if the vector id does not exist.

        Note: For any delete call, if namespace is not specified, the default namespace `""` is used.
        Since the delete operation does not error when ids are not present, this means you may not receive
        an error if you delete from the wrong namespace.

        Delete can occur in the following mutual exclusive ways:
        1. Delete by ids from a single namespace
        2. Delete all vectors from a single namespace by setting delete_all to True
        3. Delete all vectors from a single namespace by specifying a metadata filter
            (note that for this option delete all must be set to False)

        API reference: https://docs.pinecone.io/reference/delete_post

        ```python
        import asyncio
        from pinecone import Pinecone, Vector, SparseValues

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                # Delete specific ids
                await idx.delete(
                    ids=['id1', 'id2'],
                    namespace='my_namespace'
                )

                # Delete everything in a namespace
                await idx.delete(
                    delete_all=True,
                    namespace='my_namespace'
                )

                # Delete by metadata filter
                await idx.delete(
                    filter={'key': 'value'},
                    namespace='my_namespace'
                )

        asyncio.run(main())
        ```

          Returns: An empty dictionary if the delete operation was successful.
        """
        pass

    @abstractmethod
    async def fetch(
        self, ids: List[str], namespace: Optional[str] = None, **kwargs
    ) -> FetchResponse:
        """
        The fetch operation looks up and returns vectors, by ID, from a single namespace.
        The returned vectors include the vector data and/or metadata.

        API reference: https://docs.pinecone.io/reference/fetch

        ```
        import asyncio
        from pinecone import Pinecone, Vector, SparseValues

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                # Fetch specific ids in namespace
                fetched = await idx.fetch(
                    ids=['id1', 'id2'],
                    namespace='my_namespace'
                )
                for vec_id in fetched.vectors:
                    vector = fetched.vectors[vec_id]
                    print(vector.id)
                    print(vector.metadata)
                    print(vector.values)

        asyncio.run(main())
        ```

        Args:
            ids (List[str]): The vector IDs to fetch.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]

        Returns: FetchResponse object which contains the list of Vector objects, and namespace name.
        """
        pass

    @abstractmethod
    async def query(
        self,
        *args,
        top_k: int,
        vector: Optional[List[float]] = None,
        id: Optional[str] = None,
        namespace: Optional[str] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> QueryResponse:
        """
        The Query operation searches a namespace, using a query vector.
        It retrieves the ids of the most similar items in a namespace, along with their similarity scores.

        API reference: https://docs.pinecone.io/reference/query

        ## Querying with dense vectors

        ```python
        import asyncio
        from pinecone import Pinecone, Vector, SparseValues

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                query_embedding = [0.1, 0.2, 0.3, ...] # An embedding that matches the index dimension

                # Query by vector values
                results = await idx.query(
                    vector=query_embedding,
                    top_k=10,
                    filter={'genre': {"$eq": "drama"}}, # Optionally filter by metadata
                    namespace='my_namespace',
                    include_values=False,
                    include_metadata=True
                )

                # Query using vector id (the values from this stored vector will be used to query)
                results = await idx.query(
                    id='1',
                    top_k=10,
                    filter={"year": {"$gt": 2000}},
                    namespace='my_namespace',
                )

        asyncio.run(main())
        ```

        ## Query with sparse vectors

        ```python
        import asyncio
        from pinecone import Pinecone, Vector, SparseValues

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                query_embedding = [0.1, 0.2, 0.3, ...] # An embedding that matches the index dimension

                # Query by vector values
                results = await idx.query(
                    vector=query_embedding,
                    top_k=10,
                    filter={'genre': {"$eq": "drama"}}, # Optionally filter by metadata
                    namespace='my_namespace',
                    include_values=False,
                    include_metadata=True
                )

                # Query using vector id (the values from this stored vector will be used to query)
                results = await idx.query(
                    id='1',
                    top_k=10,
                    filter={"year": {"$gt": 2000}},
                    namespace='my_namespace',
                )

        asyncio.run(main())
        ```

        Examples:
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace')
            >>> index.query(id='id1', top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], top_k=10, namespace='my_namespace', filter={'key': 'value'})
            >>> index.query(id='id1', top_k=10, namespace='my_namespace', include_metadata=True, include_values=True)
            >>> index.query(vector=[1, 2, 3], sparse_vector={'indices': [1, 2], 'values': [0.2, 0.4]},
            >>>             top_k=10, namespace='my_namespace')
            >>> index.query(vector=[1, 2, 3], sparse_vector=SparseValues([1, 2], [0.2, 0.4]),
            >>>             top_k=10, namespace='my_namespace')

        Args:
            vector (List[float]): The query vector. This should be the same length as the dimension of the index
                                  being queried. Each `query()` request can contain only one of the parameters
                                  `id` or `vector`.. [optional]
            id (str): The unique ID of the vector to be used as a query vector.
                      Each `query()` request can contain only one of the parameters
                      `vector` or  `id`. [optional]
            top_k (int): The number of results to return for each query. Must be an integer greater than 1.
            namespace (str): The namespace to fetch vectors from.
                             If not specified, the default namespace is used. [optional]
            filter (Dict[str, Union[str, float, int, bool, List, dict]):
                    The filter to apply. You can use vector metadata to limit your search.
                    See https://www.pinecone.io/docs/metadata-filtering/.. [optional]
            include_values (bool): Indicates whether vector values are included in the response.
                                   If omitted the server will use the default value of False [optional]
            include_metadata (bool): Indicates whether metadata is included in the response as well as the ids.
                                     If omitted the server will use the default value of False  [optional]
            sparse_vector: (Union[SparseValues, Dict[str, Union[List[float], List[int]]]]): sparse values of the query vector.
                            Expected to be either a SparseValues object or a dict of the form:
                             {'indices': List[int], 'values': List[float]}, where the lists each have the same length.

        Returns: QueryResponse object which contains the list of the closest vectors as ScoredVector objects,
                 and namespace name.
        """
        pass

    @abstractmethod
    async def query_namespaces(
        self,
        namespaces: List[str],
        top_k: Optional[int] = None,
        filter: Optional[FilterTypedDict] = None,
        include_values: Optional[bool] = None,
        include_metadata: Optional[bool] = None,
        vector: Optional[List[float]] = None,
        sparse_vector: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> QueryNamespacesResults:
        """The query_namespaces() method is used to make a query to multiple namespaces in parallel and combine the results into one result set.

        Args:
            vector (List[float]): The query vector, must be the same length as the dimension of the index being queried.
            namespaces (List[str]): The list of namespaces to query.
            top_k (Optional[int], optional): The number of results you would like to request from each namespace. Defaults to 10.
            filter (Optional[Dict[str, Union[str, float, int, bool, List, dict]]], optional): Pass an optional filter to filter results based on metadata. Defaults to None.
            include_values (Optional[bool], optional): Boolean field indicating whether vector values should be included with results. Defaults to None.
            include_metadata (Optional[bool], optional): Boolean field indicating whether vector metadata should be included with results. Defaults to None.
            sparse_vector (Optional[ Union[SparseValues, Dict[str, Union[List[float], List[int]]]] ], optional): If you are working with a dotproduct index, you can pass a sparse vector as part of your hybrid search. Defaults to None.

        Returns:
            QueryNamespacesResults: A QueryNamespacesResults object containing the combined results from all namespaces, as well as the combined usage cost in read units.

        Examples:

        ```python
        import asyncio
        from pinecone import Pinecone

        async def main():
            pc = Pinecone(api_key="your-api-key")
            idx = pc.IndexAsyncio(
                host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io",
            )

            query_vec = [0.1, 0.2, 0.3] # An embedding that matches the index dimension
            combined_results = await idx.query_namespaces(
                vector=query_vec,
                namespaces=['ns1', 'ns2', 'ns3', 'ns4'],
                top_k=10,
                filter={'genre': {"$eq": "drama"}},
                include_values=True,
                include_metadata=True
            )
            for vec in combined_results.matches:
                print(vec.id, vec.score)
            print(combined_results.usage)

            await idx.close()

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        set_metadata: Optional[VectorMetadataTypedDict] = None,
        namespace: Optional[str] = None,
        sparse_values: Optional[Union[SparseValues, SparseVectorTypedDict]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        The Update operation updates vector in a namespace.

        Args:
            id (str): Vector's unique id.
            values (List[float]): vector values to set. [optional]
            set_metadata (Dict[str, Union[str, float, int, bool, List[int], List[float], List[str]]]]):
                metadata to set for vector. [optional]
            namespace (str): Namespace name where to update the vector.. [optional]
            sparse_values: (Dict[str, Union[List[float], List[int]]]): sparse values to update for the vector.
                           Expected to be either a SparseValues object or a dict of the form:
                           {'indices': List[int], 'values': List[float]} where the lists each have the same length.

        If a value is included, it will overwrite the previous value.
        If a set_metadata is included,
        the values of the fields specified in it will be added or overwrite the previous value.

        API reference: https://docs.pinecone.io/reference/update

        Examples:
        ```python
        import asyncio
        from pinecone import Pinecone, Vector, SparseValues

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                # Update vector values
                await idx.update(
                    id='id1',
                    values=[0.1, 0.2, 0.3, ...],
                    namespace='my_namespace'
                )

                # Update metadata
                await idx.update(
                    id='id1',
                    set_metadata={'key': 'value'},
                    namespace='my_namespace'
                )

                # Update sparse values
                await idx.update(
                    id='id1',
                    sparse_values={'indices': [1, 2], 'values': [0.2, 0.4]},
                    namespace='my_namespace'
                )

                # Update sparse values with SparseValues object
                await idx.update(
                    id='id1',
                    sparse_values=SparseValues(indices=[234781, 5432], values=[0.2, 0.4]),
                    namespace='my_namespace'
                )

        asyncio.run(main())
        ```

        """
        pass

    @abstractmethod
    async def describe_index_stats(
        self, filter: Optional[FilterTypedDict] = None, **kwargs
    ) -> DescribeIndexStatsResponse:
        """
        The DescribeIndexStats operation returns statistics about the index's contents.
        For example: The vector count per namespace and the number of dimensions.

        API reference: https://docs.pinecone.io/reference/describe_index_stats_post

        Args:
            filter (Dict[str, Union[str, float, int, bool, List, dict]]):
            If this parameter is present, the operation only returns statistics for vectors that satisfy the filter.
            See https://www.pinecone.io/docs/metadata-filtering/.. [optional]

        Returns: DescribeIndexStatsResponse object which contains stats about the index.

        ```python
        import asyncio
        from pinecone import Pinecone, Vector, SparseValues

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                print(await idx.describe_index_stats())

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        pagination_token: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs,
    ) -> ListResponse:
        """
        The list_paginated operation finds vectors based on an id prefix within a single namespace.
        It returns matching ids in a paginated form, with a pagination token to fetch the next page of results.
        This id list can then be passed to fetch or delete operations, depending on your use case.

        Consider using the `list` method to avoid having to handle pagination tokens manually.

        Examples:
            >>> results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace')
            >>> [v.id for v in results.vectors]
            ['99', '990', '991', '992', '993']
            >>> results.pagination.next
            eyJza2lwX3Bhc3QiOiI5OTMiLCJwcmVmaXgiOiI5OSJ9
            >>> next_results = index.list_paginated(prefix='99', limit=5, namespace='my_namespace', pagination_token=results.pagination.next)

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]

        Returns: ListResponse object which contains the list of ids, the namespace name, pagination information, and usage showing the number of read_units consumed.
        """
        pass

    @abstractmethod
    async def list(self, **kwargs):
        """
        The list operation accepts all of the same arguments as list_paginated, and returns a generator that yields
        a list of the matching vector ids in each page of results. It automatically handles pagination tokens on your
        behalf.

        Examples:
            >>> for ids in index.list(prefix='99', limit=5, namespace='my_namespace'):
            >>>     print(ids)
            ['99', '990', '991', '992', '993']
            ['994', '995', '996', '997', '998']
            ['999']

        Args:
            prefix (Optional[str]): The id prefix to match. If unspecified, an empty string prefix will
                                    be used with the effect of listing all ids in a namespace [optional]
            limit (Optional[int]): The maximum number of ids to return. If unspecified, the server will use a default value. [optional]
            pagination_token (Optional[str]): A token needed to fetch the next page of results. This token is returned
                in the response if additional results are available. [optional]
            namespace (Optional[str]): The namespace to fetch vectors from. If not specified, the default namespace is used. [optional]
        """
        pass

    @abstractmethod
    async def upsert_records(self, namespace: str, records: List[Dict]):
        """
        :param namespace: The namespace of the index to upsert records to.
        :type namespace: str, required
        :param records: The records to upsert into the index.
        :type records: List[Dict], required

        Upsert records to a namespace. A record is a dictionary that contains eitiher an `id` or `_id`
        field along with other fields that will be stored as metadata. The `id` or `_id` field is used
        as the unique identifier for the record. At least one field in the record should correspond to
        a field mapping in the index's embed configuration.

        When records are upserted, Pinecone converts mapped fields into embeddings and upserts them into
        the specified namespacce of the index.

        ```python
        import asyncio
        from pinecone import (
            Pinecone,
            CloudProvider,
            AwsRegion,
            EmbedModel
            IndexEmbed
        )

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                # upsert records
                await idx.upsert_records(
                    namespace="my-namespace",
                    records=[
                        {
                            "_id": "test1",
                            "my_text_field": "Apple is a popular fruit known for its sweetness and crisp texture.",
                        },
                        {
                            "_id": "test2",
                            "my_text_field": "The tech company Apple is known for its innovative products like the iPhone.",
                        },
                        {
                            "_id": "test3",
                            "my_text_field": "Many people enjoy eating apples as a healthy snack.",
                        },
                        {
                            "_id": "test4",
                            "my_text_field": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
                        },
                        {
                            "_id": "test5",
                            "my_text_field": "An apple a day keeps the doctor away, as the saying goes.",
                        },
                        {
                            "_id": "test6",
                            "my_text_field": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership.",
                        },
                    ],
                )

                from pinecone import SearchQuery, SearchRerank, RerankModel

                # search for similar records
                response = await idx.search_records(
                    namespace="my-namespace",
                    query=SearchQuery(
                        inputs={
                            "text": "Apple corporation",
                        },
                        top_k=3,
                    ),
                    rerank=SearchRerank(
                        model=RerankModel.Bge_Reranker_V2_M3,
                        rank_fields=["my_text_field"],
                        top_n=3,
                    ),
                )

        asyncio.run(main())
        ```
        """
        pass

    @abstractmethod
    async def search(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """
        :param namespace: The namespace in the index to search.
        :type namespace: str, required
        :param query: The SearchQuery to use for the search.
        :type query: Union[Dict, SearchQuery], required
        :param rerank: The SearchRerank to use with the search request.
        :type rerank: Union[Dict, SearchRerank], optional
        :return: The records that match the search.

        Search for records.

        This operation converts a query to a vector embedding and then searches a namespace. You
        can optionally provide a reranking operation as part of the search.

        ```python
        import asyncio
        from pinecone import (
            Pinecone,
            CloudProvider,
            AwsRegion,
            EmbedModel
            IndexEmbed
        )

        async def main():
            pc = Pinecone()
            async with pc.IndexAsyncio(host="example-dojoi3u.svc.aped-4627-b74a.pinecone.io") as idx:
                # upsert records
                await idx.upsert_records(
                    namespace="my-namespace",
                    records=[
                        {
                            "_id": "test1",
                            "my_text_field": "Apple is a popular fruit known for its sweetness and crisp texture.",
                        },
                        {
                            "_id": "test2",
                            "my_text_field": "The tech company Apple is known for its innovative products like the iPhone.",
                        },
                        {
                            "_id": "test3",
                            "my_text_field": "Many people enjoy eating apples as a healthy snack.",
                        },
                        {
                            "_id": "test4",
                            "my_text_field": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
                        },
                        {
                            "_id": "test5",
                            "my_text_field": "An apple a day keeps the doctor away, as the saying goes.",
                        },
                        {
                            "_id": "test6",
                            "my_text_field": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership.",
                        },
                    ],
                )

                from pinecone import SearchQuery, SearchRerank, RerankModel

                # search for similar records
                response = await idx.search_records(
                    namespace="my-namespace",
                    query=SearchQuery(
                        inputs={
                            "text": "Apple corporation",
                        },
                        top_k=3,
                    ),
                    rerank=SearchRerank(
                        model=RerankModel.Bge_Reranker_V2_M3,
                        rank_fields=["my_text_field"],
                        top_n=3,
                    ),
                )

        asyncio.run(main())
        ```

        """
        pass

    @abstractmethod
    async def search_records(
        self,
        namespace: str,
        query: Union[SearchQueryTypedDict, SearchQuery],
        rerank: Optional[Union[SearchRerankTypedDict, SearchRerank]] = None,
        fields: Optional[List[str]] = ["*"],  # Default to returning all fields
    ) -> SearchRecordsResponse:
        """Alias of the search() method."""
        pass
