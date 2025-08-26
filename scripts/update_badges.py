import re
import subprocess
import tomli
from pathlib import Path

def extract_dependency_info():
    """Extract version and commit hash for kokoro and misaki from pyproject.toml"""
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    
    deps = pyproject["project"]["dependencies"]
    info = {}
    
    # Extract kokoro info
    for dep in deps:
        if dep.startswith("kokoro @"):
            # Extract version from the dependency string if available
            version_match = re.search(r"kokoro @ git\+https://github\.com/hexgrad/kokoro\.git@", dep)
            if version_match:
                # If no explicit version, use v0.7.9 as shown in the README
                version = "v0.7.9"
            commit_match = re.search(r"@([a-f0-9]{7})", dep)
            if commit_match:
                info["kokoro"] = {
                    "version": version,
                    "commit": commit_match.group(1)
                }
        elif dep.startswith("misaki["):
            # Extract version from the dependency string if available
            version_match = re.search(r"misaki\[.*?\] @ git\+https://github\.com/hexgrad/misaki\.git@", dep)
            if version_match:
                # If no explicit version, use v0.7.9 as shown in the README
                version = "v0.7.9"
            commit_match = re.search(r"@([a-f0-9]{7})", dep)
            if commit_match:
                info["misaki"] = {
                    "version": version,
                    "commit": commit_match.group(1)
                }
    
    return info

def run_pytest_with_coverage():
    """Run pytest with coverage and return the results"""
    try:
        # Run pytest with coverage
        result = subprocess.run(
            ["pytest", "--cov=api", "-v"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract test results
        test_output = result.stdout
        passed_tests = len(re.findall(r"PASSED", test_output))
        
        # Extract coverage from .coverage file
        coverage_output = subprocess.run(
            ["coverage", "report"],
            capture_output=True,
            text=True,
            check=True
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
        r'!\[Tests\]\(https://img\.shields\.io/badge/tests-\d+%20passed-[a-zA-Z]+\)',
        f'![Tests](https://img.shields.io/badge/tests-{passed_tests}%20passed-darkgreen)',
        content
    )
    
    # Update coverage badge
    content = re.sub(
        r'!\[Coverage\]\(https://img\.shields\.io/badge/coverage-\d+%25-[a-zA-Z]+\)',
        f'![Coverage](https://img.shields.io/badge/coverage-{coverage_percentage}%25-tan)',
        content
    )
    
    # Update kokoro badge
    if "kokoro" in dep_info:
        content = re.sub(
            r'!\[Kokoro\]\(https://img\.shields\.io/badge/kokoro-[^)]+\)',
            f'![Kokoro](https://img.shields.io/badge/kokoro-{dep_info["kokoro"]["version"]}::{dep_info["kokoro"]["commit"]}-BB5420)',
            content
        )
    
    # Update misaki badge
    if "misaki" in dep_info:
        content = re.sub(
            r'!\[Misaki\]\(https://img\.shields\.io/badge/misaki-[^)]+\)',
            f'![Misaki](https://img.shields.io/badge/misaki-{dep_info["misaki"]["version"]}::{dep_info["misaki"]["commit"]}-B8860B)',
            content
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
            print(f"- Kokoro: {dep_info['kokoro']['version']}::{dep_info['kokoro']['commit']}")
        if "misaki" in dep_info:
            print(f"- Misaki: {dep_info['misaki']['version']}::{dep_info['misaki']['commit']}")
    else:
        print("Failed to update badges")

if __name__ == "__main__":
    main()