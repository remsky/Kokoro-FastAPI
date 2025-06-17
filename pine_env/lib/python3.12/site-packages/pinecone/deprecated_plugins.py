class DeprecatedPluginError(Exception):
    def __init__(self, plugin_name: str) -> None:
        message = f"The `{plugin_name}` package has been deprecated. The features from that plugin have been incorporated into the main `pinecone` package with no need for additional plugins. Please remove the `{plugin_name}` package from your dependencies to ensure you have the most up-to-date version of these features."
        super().__init__(message)


def check_for_deprecated_plugins():
    try:
        from pinecone_plugins.inference import __installables__  # type: ignore

        if __installables__ is not None:
            raise DeprecatedPluginError("pinecone-plugin-inference")
    except ImportError:
        pass

    try:
        from pinecone_plugins.records import __installables__  # type: ignore

        if __installables__ is not None:
            raise DeprecatedPluginError("pinecone-plugin-records")
    except ImportError:
        pass
