from typing import Any
from .setup_openapi_client import build_plugin_setup_client
from pinecone.config import Config
from pinecone.config.openapi_configuration import Configuration as OpenApiConfig

from pinecone_plugin_interface import load_and_install as install_plugins
import logging

logger = logging.getLogger(__name__)
""" @private """


class PluginAware:
    """
    Base class for classes that support plugin loading.

    This class provides functionality to lazily load plugins when they are first accessed.
    Subclasses must set the following attributes before calling super().__init__():
    - config: Config
    - _openapi_config: OpenApiConfig
    - _pool_threads: int

    These attributes are considered private and should not be used by end users. The config property
    is also considered private, but it was originally named without the underscore and this name
    can't be changed without breaking compatibility with plugins in the wild.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the PluginAware class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            AttributeError: If required attributes are not set in the subclass.
        """
        self._plugins_loaded = False
        """ @private """

        # Check for required attributes after super().__init__ has been called
        missing_attrs = []
        if not hasattr(self, "config"):
            missing_attrs.append("config")
        if not hasattr(self, "_openapi_config"):
            missing_attrs.append("_openapi_config")
        if not hasattr(self, "_pool_threads"):
            missing_attrs.append("_pool_threads")

        if missing_attrs:
            logger.error(
                f"PluginAware class requires the following attributes: {', '.join(missing_attrs)}. "
                f"These must be set in the {self.__class__.__name__} class's __init__ method "
                f"before calling super().__init__()."
            )
            raise AttributeError(
                f"PluginAware class requires the following attributes: {', '.join(missing_attrs)}. "
                f"These must be set in the {self.__class__.__name__} class's __init__ method "
                f"before calling super().__init__()."
            )

    def __getattr__(self, name: str) -> Any:
        """
        Called when an attribute is not found through the normal lookup process.
        This allows for lazy loading of plugins when they are first accessed.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The requested attribute.

        Raises:
            AttributeError: If the attribute cannot be found after loading plugins.
        """
        logger.debug("__getattr__ called for %s", name)
        # Check if this is one of the required attributes that should be set by subclasses
        required_attrs = ["config", "_openapi_config", "_pool_threads"]
        if name in required_attrs:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                f"This attribute must be set in the subclass's __init__ method "
                f"before calling super().__init__()."
            )

        if not self._plugins_loaded:
            logger.debug("Loading plugins for %s", self.__class__.__name__)
            # Use object.__getattribute__ to avoid triggering __getattr__ again
            try:
                config = object.__getattribute__(self, "config")
                openapi_config = object.__getattribute__(self, "_openapi_config")
                pool_threads = object.__getattribute__(self, "_pool_threads")
                self.load_plugins(
                    config=config, openapi_config=openapi_config, pool_threads=pool_threads
                )
                self._plugins_loaded = True
                try:
                    return object.__getattribute__(self, name)
                except AttributeError:
                    pass
            except AttributeError:
                # If we can't get the required attributes, we can't load plugins
                pass

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def load_plugins(
        self, config: Config, openapi_config: OpenApiConfig, pool_threads: int
    ) -> None:
        """
        Load plugins for the parent class.

        Args:
            config: The Pinecone configuration.
            openapi_config: The OpenAPI configuration.
            pool_threads: The number of threads in the pool.
        """
        try:
            # Build the OpenAPI client for plugin setup
            openapi_client_builder = build_plugin_setup_client(
                config=config, openapi_config=openapi_config, pool_threads=pool_threads
            )
            # Install plugins
            install_plugins(self, openapi_client_builder)
            logger.debug("Plugins loaded successfully for %s", self.__class__.__name__)
        except ImportError as e:
            logger.warning("Failed to import plugin module: %s", e)
        except Exception as e:
            logger.error("Error loading plugins: %s", e, exc_info=True)
