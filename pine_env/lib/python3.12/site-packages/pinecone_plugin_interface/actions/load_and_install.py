from .constants import NAMESPACE_PACKAGE
from .discover_plugins import discover_plugins
from .discover_namespace_packages import discover_subpackages
from .installation import install_plugins

def load_and_install(target, plugin_client_builder):
    packages = discover_subpackages(NAMESPACE_PACKAGE)
    plugins = discover_plugins(target.__class__.__name__, packages)
    install_plugins(target, plugins, plugin_client_builder)