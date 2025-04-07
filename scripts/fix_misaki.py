"""
Patch for misaki package to fix the EspeakWrapper.set_data_path issue.
"""

import importlib.util
import os
import sys

# Find the misaki package
try:
    import misaki

    misaki_path = os.path.dirname(misaki.__file__)
    print(f"Found misaki package at: {misaki_path}")
except ImportError:
    print("Misaki package not found. Make sure it's installed.")
    sys.exit(1)

# Path to the espeak.py file
espeak_file = os.path.join(misaki_path, "espeak.py")

if not os.path.exists(espeak_file):
    print(f"Could not find {espeak_file}")
    sys.exit(1)

# Read the current content
with open(espeak_file, "r") as f:
    content = f.read()

# Check if the problematic line exists
if "EspeakWrapper.set_data_path(espeakng_loader.get_data_path())" in content:
    # Replace the problematic line
    new_content = content.replace(
        "EspeakWrapper.set_data_path(espeakng_loader.get_data_path())",
        "# Fixed line to use data_path attribute instead of set_data_path method\n"
        "EspeakWrapper.data_path = espeakng_loader.get_data_path()",
    )

    # Write the modified content back
    with open(espeak_file, "w") as f:
        f.write(new_content)

    print(f"Successfully patched {espeak_file}")
else:
    print(f"The problematic line was not found in {espeak_file}")
    print("The file may have already been patched or the issue is different.")
