class IndexClientInstantiationError(Exception):
    def __init__(self, index_args, index_kwargs) -> None:
        formatted_args = ", ".join(map(repr, index_args))
        formatted_kwargs = ", ".join(f"{key}={repr(value)}" for key, value in index_kwargs.items())
        combined_args = ", ".join([a for a in [formatted_args, formatted_kwargs] if a.strip()])

        self.message = f"""You are attempting to access the Index client directly from the pinecone module. The Index client must be instantiated through the parent Pinecone client instance so that it can inherit shared configurations such as API key.

    INCORRECT USAGE:
        ```
        import pinecone

        pc = pinecone.Pinecone(api_key='your-api-key')
        index = pinecone.Index({combined_args})
        ```

    CORRECT USAGE:
        ```
        from pinecone import Pinecone

        pc = Pinecone(api_key='your-api-key')
        index = pc.Index({combined_args})
        ```
        """
        super().__init__(self.message)


class InferenceInstantiationError(Exception):
    def __init__(self) -> None:
        self.message = """You are attempting to access the Inference client directly from the pinecone module. Inference functionality such as `embed` and `rerank` should only be accessed through the parent Pinecone client instance.

    INCORRECT USAGE:
        ```
        import pinecone

        pinecone.Inference().embed(...)
        ```

    CORRECT USAGE:
        ```
        from pinecone import Pinecone

        pc = Pinecone(api_key='your-api-key')

        embeddings = pc.inference.embed(
            model='multilingual-e5-large',
            inputs=["The quick brown fox jumps over the lazy dog.", "lorem ipsum"],
            parameters={"input_type": "query", "truncate": "END"},
        )
        ```
        """
        super().__init__(self.message)


class Index:
    def __init__(self, *args, **kwargs) -> None:
        raise IndexClientInstantiationError(args, kwargs)


class Inference:
    def __init__(self, *args, **kwargs) -> None:
        raise InferenceInstantiationError()
