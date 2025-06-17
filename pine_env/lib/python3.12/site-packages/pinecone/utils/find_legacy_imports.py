#!/usr/bin/env python3
"""
Script to identify legacy imports that were previously available via star imports.

This script analyzes the codebase to find all imports that were previously available
via star imports but are no longer imported at the top level.
"""

import ast
import os
from typing import Set


def find_star_imports(file_path: str) -> Set[str]:
    """
    Find all star imports in a file.

    Args:
        file_path: Path to the file to analyze.

    Returns:
        Set of module names that are imported with star imports.
    """
    with open(file_path, "r") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        print(f"Warning: Could not parse {file_path}")
        return set()

    star_imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.names[0].name == "*":
            module_name = node.module
            if module_name:
                star_imports.add(module_name)

    return star_imports


def find_imported_names(file_path: str) -> Set[str]:
    """
    Find all names that are imported in a file.

    Args:
        file_path: Path to the file to analyze.

    Returns:
        Set of imported names.
    """
    with open(file_path, "r") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        print(f"Warning: Could not parse {file_path}")
        return set()

    imported_names = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imported_names.add(name.name)
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                if name.name != "*":
                    imported_names.add(name.name)

    return imported_names


def find_module_exports(module_path: str) -> Set[str]:
    """
    Find all names that are exported by a module.

    Args:
        module_path: Path to the module to analyze.

    Returns:
        Set of exported names.
    """
    try:
        module = __import__(module_path, fromlist=["*"])
        return set(dir(module))
    except ImportError:
        print(f"Warning: Could not import {module_path}")
        return set()


def main():
    """
    Main function to find legacy imports.
    """
    # Get the package root directory
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Find the __init__.py file
    init_file = os.path.join(package_root, "__init__.py")

    # Find star imports in the __init__.py file
    star_imports = find_star_imports(init_file)

    # Find all imported names in the __init__.py file
    imported_names = find_imported_names(init_file)

    # Find all module exports
    module_exports = {}
    for module_name in star_imports:
        module_exports[module_name] = find_module_exports(module_name)

    # Find all files in the package
    package_files = []
    for root, _, files in os.walk(package_root):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                package_files.append(os.path.join(root, file))

    # Find all imports in the package
    package_imports = set()
    for file in package_files:
        package_imports.update(find_imported_names(file))

    # Find legacy imports
    legacy_imports = {}
    for module_name, exports in module_exports.items():
        for export in exports:
            if export in package_imports and export not in imported_names:
                legacy_imports[f"pinecone.{export}"] = (module_name, export)

    # Print the legacy imports
    print("LEGACY_IMPORTS = {")
    for legacy_name, (module_path, actual_name) in sorted(legacy_imports.items()):
        print(f"    '{legacy_name}': ('{module_path}', '{actual_name}'),")
    print("}")


if __name__ == "__main__":
    main()
