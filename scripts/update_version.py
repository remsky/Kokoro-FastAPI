#!/usr/bin/env python3
"""
Version Update Script

This script reads the version from the VERSION file and updates references
in pyproject.toml, the Helm chart, and README.md.
"""

import re
from pathlib import Path

import yaml

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent

# --- Configuration ---
VERSION_FILE = ROOT_DIR / "VERSION"
PYPROJECT_FILE = ROOT_DIR / "pyproject.toml"
HELM_CHART_FILE = ROOT_DIR / "charts" / "kokoro-fastapi" / "Chart.yaml"
README_FILE = ROOT_DIR / "README.md"
# --- End Configuration ---


def update_pyproject(version: str):
    """Updates the version in pyproject.toml"""
    if not PYPROJECT_FILE.exists():
        print(f"Skipping: {PYPROJECT_FILE} not found.")
        return

    try:
        content = PYPROJECT_FILE.read_text()
        # Regex to find and capture current version = "X.Y.Z" under [project]
        pattern = r'(^\[project\]\s*(?:.*\s)*?version\s*=\s*)"([^"]+)"'
        match = re.search(pattern, content, flags=re.MULTILINE)

        if not match:
            print(f"Warning: Version pattern not found in {PYPROJECT_FILE}")
            return

        current_version = match.group(2)
        if current_version == version:
            print(f"Already up-to-date: {PYPROJECT_FILE} (version {version})")
        else:
            # Perform replacement
            new_content = re.sub(
                pattern, rf'\1"{version}"', content, count=1, flags=re.MULTILINE
            )
            PYPROJECT_FILE.write_text(new_content)
            print(f"Updated {PYPROJECT_FILE} from {current_version} to {version}")

    except Exception as e:
        print(f"Error processing {PYPROJECT_FILE}: {e}")


def update_helm_chart(version: str):
    """Updates the version and appVersion in the Helm chart"""
    if not HELM_CHART_FILE.exists():
        print(f"Skipping: {HELM_CHART_FILE} not found.")
        return

    try:
        content = HELM_CHART_FILE.read_text()
        original_content = content
        updated_count = 0

        # Update 'version:' line (unquoted)
        # Looks for 'version:' followed by optional whitespace and the version number
        version_pattern = r"^(version:\s*)(\S+)"
        current_version_match = re.search(version_pattern, content, flags=re.MULTILINE)
        if current_version_match and current_version_match.group(2) != version:
            content = re.sub(
                version_pattern,
                rf"\g<1>{version}",
                content,
                count=1,
                flags=re.MULTILINE,
            )
            print(
                f"Updating 'version' in {HELM_CHART_FILE} from {current_version_match.group(2)} to {version}"
            )
            updated_count += 1
        elif current_version_match:
            print(f"Already up-to-date: 'version' in {HELM_CHART_FILE} is {version}")
        else:
            print(f"Warning: 'version:' pattern not found in {HELM_CHART_FILE}")

        # Update 'appVersion:' line (quoted or unquoted)
        # Looks for 'appVersion:' followed by optional whitespace, optional quote, the version, optional quote
        app_version_pattern = r"^(appVersion:\s*)(\"?)([^\"\s]+)(\"?)"
        current_app_version_match = re.search(
            app_version_pattern, content, flags=re.MULTILINE
        )

        if current_app_version_match:
            leading_whitespace = current_app_version_match.group(
                1
            )  # e.g., "appVersion: "
            opening_quote = current_app_version_match.group(2)  # e.g., '"' or ''
            current_app_ver = current_app_version_match.group(3)  # e.g., '0.2.0'
            closing_quote = current_app_version_match.group(4)  # e.g., '"' or ''

            # Check if quotes were consistent (both present or both absent)
            if opening_quote != closing_quote:
                print(
                    f"Warning: Inconsistent quotes found for appVersion in {HELM_CHART_FILE}. Skipping update for this line."
                )
            elif (
                current_app_ver == version and opening_quote == '"'
            ):  # Check if already correct *and* quoted
                print(
                    f"Already up-to-date: 'appVersion' in {HELM_CHART_FILE} is \"{version}\""
                )
            else:
                # Always replace with the quoted version
                replacement = f'{leading_whitespace}"{version}"'  # Ensure quotes
                original_display = f"{opening_quote}{current_app_ver}{closing_quote}"  # How it looked before
                target_display = f'"{version}"'  # How it should look

                # Only report update if the displayed value actually changes
                if original_display != target_display:
                    content = re.sub(
                        app_version_pattern,
                        replacement,
                        content,
                        count=1,
                        flags=re.MULTILINE,
                    )
                    print(
                        f"Updating 'appVersion' in {HELM_CHART_FILE} from {original_display} to {target_display}"
                    )
                    updated_count += 1
                else:
                    # It matches the target version but might need quoting fixed silently if we didn't update
                    # Or it was already correct. Check if content changed. If not, report up-to-date.
                    if not (
                        content != original_content and updated_count > 0
                    ):  # Avoid double message if version also changed
                        print(
                            f"Already up-to-date: 'appVersion' in {HELM_CHART_FILE} is {target_display}"
                        )

        else:
            print(f"Warning: 'appVersion:' pattern not found in {HELM_CHART_FILE}")

        # Write back only if changes were made
        if content != original_content:
            HELM_CHART_FILE.write_text(content)
            # Confirmation message printed above during the specific update
        elif updated_count == 0 and current_version_match and current_app_version_match:
            # If no updates were made but patterns were found, confirm it's up-to-date overall
            print(f"Already up-to-date: {HELM_CHART_FILE} (version {version})")

    except Exception as e:
        print(f"Error processing {HELM_CHART_FILE}: {e}")


def update_readme(version_with_v: str):
    """Updates Docker image tags in README.md"""
    if not README_FILE.exists():
        print(f"Skipping: {README_FILE} not found.")
        return

    try:
        content = README_FILE.read_text()
        # Regex to find and capture current ghcr.io/.../kokoro-fastapi-(cpu|gpu):vX.Y.Z
        pattern = r"(ghcr\.io/remsky/kokoro-fastapi-(?:cpu|gpu)):(v\d+\.\d+\.\d+)"
        matches = list(re.finditer(pattern, content))  # Find all occurrences

        if not matches:
            print(f"Warning: Docker image tag pattern not found in {README_FILE}")
        else:
            updated_needed = False
            for match in matches:
                current_tag = match.group(2)
                if current_tag != version_with_v:
                    updated_needed = True
                    break  # Only need one mismatch to trigger update

            if updated_needed:
                # Perform replacement on all occurrences
                new_content = re.sub(pattern, rf"\1:{version_with_v}", content)
                README_FILE.write_text(new_content)
                print(f"Updated Docker image tags in {README_FILE} to {version_with_v}")
            else:
                print(
                    f"Already up-to-date: Docker image tags in {README_FILE} (version {version_with_v})"
                )

        # Check for ':latest' tag usage remains the same
        if ":latest" in content:
            print(
                f"Warning: Found ':latest' tag in {README_FILE}. Consider updating manually if needed."
            )

    except Exception as e:
        print(f"Error processing {README_FILE}: {e}")


def main():
    # Read the version from the VERSION file
    if not VERSION_FILE.exists():
        print(f"Error: {VERSION_FILE} not found.")
        return

    try:
        version = VERSION_FILE.read_text().strip()
        if not re.match(r"^\d+\.\d+\.\d+$", version):
            print(
                f"Error: Invalid version format '{version}' in {VERSION_FILE}. Expected X.Y.Z"
            )
            return
    except Exception as e:
        print(f"Error reading {VERSION_FILE}: {e}")
        return

    print(f"Read version: {version} from {VERSION_FILE}")
    print("-" * 20)

    # Prepare versions (with and without 'v')
    version_plain = version
    version_with_v = f"v{version}"

    # Update files
    update_pyproject(version_plain)
    update_helm_chart(version_plain)
    update_readme(version_with_v)

    print("-" * 20)
    print("Version update script finished.")


if __name__ == "__main__":
    main()
