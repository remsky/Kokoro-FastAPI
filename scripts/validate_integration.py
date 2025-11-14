#!/usr/bin/env python3
"""
Validation script for ZipVoice integration.
Checks file structure, syntax, and dependencies without requiring full installation.
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def log_info(msg: str):
    print(f"{BLUE}[INFO]{RESET} {msg}")

def log_success(msg: str):
    print(f"{GREEN}[✓]{RESET} {msg}")

def log_error(msg: str):
    print(f"{RED}[✗]{RESET} {msg}")

def log_warning(msg: str):
    print(f"{YELLOW}[!]{RESET} {msg}")


class IntegrationValidator:
    """Validates ZipVoice integration without requiring imports."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.errors = []
        self.warnings = []
        self.successes = []

    def validate_all(self) -> bool:
        """Run all validations."""
        log_info("Starting ZipVoice integration validation...")
        print()

        # Run all checks
        self.check_file_structure()
        self.check_python_syntax()
        self.check_imports_structure()
        self.check_router_integration()
        self.check_docker_setup()
        self.check_documentation()

        # Print summary
        print()
        print("=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"{GREEN}Successes: {len(self.successes)}{RESET}")
        print(f"{YELLOW}Warnings: {len(self.warnings)}{RESET}")
        print(f"{RED}Errors: {len(self.errors)}{RESET}")
        print()

        if self.errors:
            log_error("Validation FAILED - errors found:")
            for error in self.errors:
                print(f"  - {error}")
            return False
        elif self.warnings:
            log_warning("Validation passed with warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
            return True
        else:
            log_success("All validations PASSED!")
            return True

    def check_file_structure(self):
        """Check that all required files exist."""
        log_info("Checking file structure...")

        required_files = [
            # Core backend files
            "api/src/inference/zipvoice.py",
            "api/src/inference/optimized_zipvoice.py",
            "api/src/inference/voice_prompt_manager.py",

            # Smart services
            "api/src/services/auto_transcription.py",
            "api/src/services/quality_detection.py",
            "api/src/services/smart_tuning.py",

            # Routers
            "api/src/routers/zipvoice_enhanced.py",

            # Config
            "api/src/core/config.py",

            # Docker
            "docker/gpu/Dockerfile",
            "docker/cpu/Dockerfile",
            "docker/gpu/docker-compose.yml",
            "docker/cpu/docker-compose.yml",
            "docker/scripts/entrypoint-zipvoice.sh",
            "docker/scripts/healthcheck.sh",
            "docker/README.md",

            # Docs
            "README.md",
            "pyproject.toml",
        ]

        for file_path in required_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                self.successes.append(f"File exists: {file_path}")
            else:
                self.errors.append(f"Missing file: {file_path}")

        log_success(f"Checked {len(required_files)} required files")

    def check_python_syntax(self):
        """Check Python files for syntax errors."""
        log_info("Checking Python syntax...")

        python_files = [
            "api/src/inference/zipvoice.py",
            "api/src/inference/optimized_zipvoice.py",
            "api/src/inference/voice_prompt_manager.py",
            "api/src/services/auto_transcription.py",
            "api/src/services/quality_detection.py",
            "api/src/services/smart_tuning.py",
            "api/src/routers/zipvoice_enhanced.py",
        ]

        syntax_ok = 0
        for file_path in python_files:
            full_path = self.root_dir / file_path
            if not full_path.exists():
                continue

            try:
                with open(full_path, 'r') as f:
                    ast.parse(f.read())
                syntax_ok += 1
            except SyntaxError as e:
                self.errors.append(f"Syntax error in {file_path}: {e}")

        if syntax_ok == len(python_files):
            log_success(f"All {syntax_ok} Python files have valid syntax")
            self.successes.append("Python syntax validation passed")
        else:
            log_error(f"Syntax errors found in {len(python_files) - syntax_ok} files")

    def check_imports_structure(self):
        """Check that imports are structured correctly."""
        log_info("Checking import structure...")

        # Check main.py imports the enhanced router
        main_py = self.root_dir / "api/src/main.py"
        if main_py.exists():
            content = main_py.read_text()
            if "zipvoice_enhanced" in content:
                log_success("main.py imports zipvoice_enhanced router")
                self.successes.append("Enhanced router imported in main.py")
            else:
                self.warnings.append("zipvoice_enhanced not found in main.py imports")

        # Check config has ZipVoice settings
        config_py = self.root_dir / "api/src/core/config.py"
        if config_py.exists():
            content = config_py.read_text()
            zipvoice_settings = [
                "enable_zipvoice",
                "zipvoice_model",
                "enable_auto_transcription",
                "enable_smart_tuning",
                "enable_quality_detection",
            ]
            found = sum(1 for s in zipvoice_settings if s in content)
            if found == len(zipvoice_settings):
                log_success(f"All {len(zipvoice_settings)} ZipVoice settings in config")
                self.successes.append("Config has all ZipVoice settings")
            else:
                self.warnings.append(f"Only {found}/{len(zipvoice_settings)} ZipVoice settings found in config")

    def check_router_integration(self):
        """Check that routers are properly integrated."""
        log_info("Checking router integration...")

        main_py = self.root_dir / "api/src/main.py"
        if not main_py.exists():
            self.errors.append("main.py not found")
            return

        content = main_py.read_text()

        # Check router import
        if "from .routers.zipvoice_enhanced import router as zipvoice_router" in content:
            log_success("ZipVoice router import found")
            self.successes.append("Router properly imported")
        else:
            self.errors.append("ZipVoice router not imported in main.py")

        # Check router inclusion
        if 'app.include_router(zipvoice_router' in content:
            log_success("ZipVoice router included in app")
            self.successes.append("Router included in FastAPI app")
        else:
            self.warnings.append("Router may not be included in FastAPI app")

    def check_docker_setup(self):
        """Check Docker configuration."""
        log_info("Checking Docker setup...")

        # Check GPU Dockerfile
        gpu_dockerfile = self.root_dir / "docker/gpu/Dockerfile"
        if gpu_dockerfile.exists():
            content = gpu_dockerfile.read_text()
            if "zipvoice" in content.lower() and "ENABLE_ZIPVOICE=true" in content:
                log_success("GPU Dockerfile configured for ZipVoice")
                self.successes.append("GPU Dockerfile properly configured")
            else:
                self.warnings.append("GPU Dockerfile may not have ZipVoice configuration")

        # Check CPU Dockerfile
        cpu_dockerfile = self.root_dir / "docker/cpu/Dockerfile"
        if cpu_dockerfile.exists():
            content = cpu_dockerfile.read_text()
            if "zipvoice" in content.lower() and "ENABLE_ZIPVOICE=true" in content:
                log_success("CPU Dockerfile configured for ZipVoice")
                self.successes.append("CPU Dockerfile properly configured")
            else:
                self.warnings.append("CPU Dockerfile may not have ZipVoice configuration")

        # Check docker-compose files
        for compose_file in ["docker/gpu/docker-compose.yml", "docker/cpu/docker-compose.yml"]:
            compose_path = self.root_dir / compose_file
            if compose_path.exists():
                content = compose_path.read_text()
                if "zipvoice" in content.lower() and "ENABLE_ZIPVOICE" in content:
                    log_success(f"{compose_file} configured")
                    self.successes.append(f"{compose_file} properly configured")
                else:
                    self.warnings.append(f"{compose_file} may be missing ZipVoice config")

        # Check scripts
        for script in ["docker/scripts/entrypoint-zipvoice.sh", "docker/scripts/healthcheck.sh"]:
            script_path = self.root_dir / script
            if script_path.exists() and script_path.stat().st_mode & 0o111:
                log_success(f"{script} exists and is executable")
                self.successes.append(f"{script} ready")
            else:
                self.errors.append(f"{script} missing or not executable")

    def check_documentation(self):
        """Check documentation files."""
        log_info("Checking documentation...")

        # Check main README
        readme = self.root_dir / "README.md"
        if readme.exists():
            content = readme.read_text()
            required_sections = ["ZipVoice", "Smart Features", "Docker"]
            found = sum(1 for s in required_sections if s in content)
            if found == len(required_sections):
                log_success("README has all required sections")
                self.successes.append("README properly updated")
            else:
                self.warnings.append(f"README missing {len(required_sections) - found} sections")

        # Check Docker README
        docker_readme = self.root_dir / "docker/README.md"
        if docker_readme.exists():
            content = docker_readme.read_text()
            if len(content) > 1000 and "ZipVoice" in content:
                log_success("Docker README is comprehensive")
                self.successes.append("Docker documentation complete")
            else:
                self.warnings.append("Docker README may be incomplete")

        # Check pyproject.toml
        pyproject = self.root_dir / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "zipvoice-fastapi" in content and "zipvoice =" in content:
                log_success("pyproject.toml updated for ZipVoice")
                self.successes.append("pyproject.toml properly configured")
            else:
                self.warnings.append("pyproject.toml may need updating")


def main():
    """Main entry point."""
    root_dir = Path(__file__).parent.parent
    validator = IntegrationValidator(root_dir)

    success = validator.validate_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
