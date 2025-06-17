"""
Legacy import handler for Pinecone.

This module provides a simple way to handle legacy imports that were previously
available via star imports but are no longer imported at the top level.
"""

import importlib
import sys
from types import ModuleType
from typing import Dict, Optional, Set, Any, Tuple, cast

# Dictionary mapping legacy import names to their actual module paths
# Format: 'name': ('module_path', 'actual_name')
LEGACY_IMPORTS: Dict[str, Tuple[str, str]] = {
    # Example: 'Vector': ('pinecone.db_data.models', 'Vector')
    # Add all your legacy imports here
}


class LegacyImportProxy:
    """
    A proxy module that handles legacy imports with warnings.

    This class is used to replace the pinecone module in sys.modules
    to handle legacy imports that were previously available via star imports.
    """

    def __init__(self, original_module: Any, legacy_imports: Dict[str, Tuple[str, str]]):
        """
        Initialize the proxy module.

        Args:
            original_module: The original module to proxy.
            legacy_imports: Dictionary of legacy imports to handle.
        """
        self._original_module = original_module
        self._legacy_imports = legacy_imports
        self._warned_imports: Set[str] = set()
        self._loaded_modules: Dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """
        Handle attribute access for legacy imports.

        Args:
            name: The name of the attribute being accessed.

        Returns:
            The requested attribute.

        Raises:
            AttributeError: If the attribute cannot be found.
        """
        # First, try to get the attribute from the original module
        try:
            return getattr(self._original_module, name)
        except AttributeError:
            pass

        # Check if this is a legacy import
        if name in self._legacy_imports:
            module_path, actual_name = self._legacy_imports[name]

            # Only warn once per import
            # if name not in self._warned_imports:
            #     warnings.warn(
            #         f"Importing '{name}' directly from 'pinecone' is deprecated. "
            #         f"Please import it from '{module_path}' instead. "
            #         f"This import will be removed in a future version.",
            #         DeprecationWarning,
            #         stacklevel=2
            #     )
            #     self._warned_imports.add(name)

            # Load the module if not already loaded
            if module_path not in self._loaded_modules:
                try:
                    self._loaded_modules[module_path] = importlib.import_module(module_path)
                except ImportError:
                    raise AttributeError(f"module 'pinecone' has no attribute '{name}'")

            # Get the actual object
            module = self._loaded_modules[module_path]
            if hasattr(module, actual_name):
                return getattr(module, actual_name)

        raise AttributeError(f"module 'pinecone' has no attribute '{name}'")


def setup_legacy_imports(legacy_imports: Optional[Dict[str, Tuple[str, str]]] = None) -> None:
    """
    Set up the legacy import handler.

    Args:
        legacy_imports: Optional dictionary of legacy imports to handle.
                       If None, uses the default LEGACY_IMPORTS dictionary.
    """
    if legacy_imports is None:
        legacy_imports = LEGACY_IMPORTS

    # Only proceed if the pinecone module is already loaded
    if "pinecone" not in sys.modules:
        return

    # Create a proxy for the pinecone module
    original_module = sys.modules["pinecone"]
    proxy = LegacyImportProxy(original_module, legacy_imports)

    # Replace the pinecone module with our proxy
    # Use a type cast to satisfy the type checker
    sys.modules["pinecone"] = cast(ModuleType, proxy)
