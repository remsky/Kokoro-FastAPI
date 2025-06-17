import importlib
from .constants import NAMESPACE_PACKAGE
from ..plugin_metadata import PluginMetadata
from ..logging import logger

def discover_plugins(target_name, subpackages):
    plugins = []
    for package in subpackages:
        logger.info(f"Looking for plugins in {package}")
        try:
            plugin_module = importlib.import_module(f"{package}")
            if not hasattr(plugin_module, '__installables__'):
                logger.exception(f"{package} does not have __installables__")
                raise Exception(f"{package} contains the {NAMESPACE_PACKAGE} namespace package but does not have __installables__")
            else:
                for plugin in plugin_module.__installables__:
                    if not isinstance(plugin, PluginMetadata):
                        logger.exception(f"installable item in {package}.__installables__ is not of type PluginMetadata")
                        raise Exception(f"installable item in {package}.__installables__ is not of type PluginMetadata")
                    else:
                        if plugin.target_object == target_name:
                            plugins.append(plugin)
        except ModuleNotFoundError:
            logger.exception(f"Something went wrong while looking for plugins in '{package}'")
            continue

    return plugins
