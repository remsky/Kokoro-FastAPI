import re
import subprocess
from pathlib import Path

import tomli


def extract_dependency_info():
    """Extract version for kokoro and misaki from pyproject.toml"""
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)

    deps = pyproject["project"]["dependencies"]
    info = {}
    kokoro_found = False
    misaki_found = False

    for dep in deps:
        # Match kokoro==version
        kokoro_match = re.match(r"^kokoro==(.+)$", dep)
        if kokoro_match:
            info["kokoro"] = {"version": kokoro_match.group(1)}
            kokoro_found = True

        # Match misaki[...] ==version or misaki==version
        misaki_match = re.match(r"^misaki(?:\[.*?\])?==(.+)$", dep)
        if misaki_match:
            info["misaki"] = {"version": misaki_match.group(1)}
            misaki_found = True

        # Stop if both found
        if kokoro_found and misaki_found:
            break

    if not kokoro_found:
        raise ValueError("Kokoro version not found in pyproject.toml dependencies")
    if not misaki_found:
        raise ValueError("Misaki version not found in pyproject.toml dependencies")

    return info


def run_pytest_with_coverage():
    """Run pytest with coverage and return the results"""
    try:
        # Run pytest with coverage
        result = subprocess.run(
            ["pytest", "--cov=api", "-v"], capture_output=True, text=True, check=True
        )

        # Extract test results
        test_output = result.stdout
        passed_tests = len(re.findall(r"PASSED", test_output))

        # Extract coverage from .coverage file
        coverage_output = subprocess.run(
            ["coverage", "report"], capture_output=True, text=True, check=True
        ).stdout

        # Extract total coverage percentage
        coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", coverage_output)
        coverage_percentage = coverage_match.group(1) if coverage_match else "0"

        return passed_tests, coverage_percentage
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        print(f"Output: {e.output}")
        return 0, "0"


def update_readme_badges(passed_tests, coverage_percentage, dep_info):
    """Update the badges in the README file"""
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("README.md not found")
        return False

    content = readme_path.read_text()

    # Update tests badge
    content = re.sub(
        r"!\[Tests\]\(https://img\.shields\.io/badge/tests-\d+%20passed-[a-zA-Z]+\)",
        f"![Tests](https://img.shields.io/badge/tests-{passed_tests}%20passed-darkgreen)",
        content,
    )

    # Update coverage badge
    content = re.sub(
        r"!\[Coverage\]\(https://img\.shields\.io/badge/coverage-\d+%25-[a-zA-Z]+\)",
        f"![Coverage](https://img.shields.io/badge/coverage-{coverage_percentage}%25-tan)",
        content,
    )

    # Update kokoro badge
    if "kokoro" in dep_info:
        # Find badge like kokoro-v0.9.2::abcdefg-BB5420 or kokoro-v0.9.2-BB5420
        kokoro_version = dep_info["kokoro"]["version"]
        content = re.sub(
            r"(!\[Kokoro\]\(https://img\.shields\.io/badge/kokoro-)[^)-]+(-BB5420\))",
            lambda m: f"{m.group(1)}{kokoro_version}{m.group(2)}",
            content,
        )

    # Update misaki badge
    if "misaki" in dep_info:
        # Find badge like misaki-v0.9.3::abcdefg-B8860B or misaki-v0.9.3-B8860B
        misaki_version = dep_info["misaki"]["version"]
        content = re.sub(
            r"(!\[Misaki\]\(https://img\.shields\.io/badge/misaki-)[^)-]+(-B8860B\))",
            lambda m: f"{m.group(1)}{misaki_version}{m.group(2)}",
            content,
        )

    readme_path.write_text(content)
    return True


def main():
    # Get dependency info
    dep_info = extract_dependency_info()

    # Run tests and get coverage
    passed_tests, coverage_percentage = run_pytest_with_coverage()

    # Update badges
    if update_readme_badges(passed_tests, coverage_percentage, dep_info):
        print(f"Updated badges:")
        print(f"- Tests: {passed_tests} passed")
        print(f"- Coverage: {coverage_percentage}%")
        if "kokoro" in dep_info:
            print(f"- Kokoro: {dep_info['kokoro']['version']}")
        if "misaki" in dep_info:
            print(f"- Misaki: {dep_info['misaki']['version']}")
    else:
        print("Failed to update badges")


if __name__ == "__main__":
    main()
