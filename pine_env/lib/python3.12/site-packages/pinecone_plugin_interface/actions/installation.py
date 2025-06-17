from typing import List
from ..plugin_metadata import PluginMetadata
from ..logging import logger

def install_plugins(target, plugins: List[PluginMetadata], plugin_client_builder):
    for plugin in plugins:
        if not isinstance(plugin, PluginMetadata):
            raise Exception("object must be an instance of PluginMetadata")
        try:
            logger.info(f"Installing plugin {plugin.namespace} into {target.__class__.__name__}")
            
            impl = plugin.implementation_class
            setattr(target, plugin.namespace, impl(target.config, plugin_client_builder))
        except Exception as e:
            # We want to display some troubleshooting information but not interrupt
            # execution of the main program in a way that would prevent non-plugin
            # related functionality from working when a broken plugin is present.
            logger.exception(f"Error while installing plugin {plugin.namespace}: {e}")
            continue