import pkgutil
import importlib
from ..logging import logger

def discover_subpackages(namespace_package):
    try:
        package = importlib.import_module(namespace_package)
    except:
        return []

    package_paths = getattr(package, '__path__', [])
    logger.info(f"Discovering subpackages in {package_paths}")

    subpackages = []
    for finder, name, ispkg in pkgutil.iter_modules(package_paths):
        if ispkg:
            full_package_name = f"{namespace_package}.{name}"
            subpackages.append(full_package_name)
    
    return subpackages