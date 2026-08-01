#!/usr/bin/env python3
"""Download and prepare Kokoro v1.0 model."""

import hashlib
import json
import os
import sys
from urllib.request import urlretrieve

# sha256 of the release assets, update alongside base_url
EXPECTED_SHA256 = {
    "kokoro-v1_0.pth": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
    "config.json": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _Logger:
    def info(self, msg: str) -> None:
        _log(msg)

    def error(self, msg: str) -> None:
        _log(f"ERROR: {msg}")


logger = _Logger()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(model_path: str, config_path: str) -> bool:
    """Verify that model files exist and are valid.

    Args:
        model_path: Path to model file
        config_path: Path to config file

    Returns:
        True if files exist and are valid
    """
    try:
        # Check files exist
        if not os.path.exists(model_path):
            return False
        if not os.path.exists(config_path):
            return False

        # Verify config file is valid JSON
        with open(config_path, encoding="utf-8") as f:
            json.load(f)

        # rejects error-page bodies saved in lieu of model (#301)
        if os.path.getsize(model_path) < 100 * 1024 * 1024:
            return False

        return True
    except Exception:
        return False


def download_model(output_dir: str) -> None:
    """Download model files from GitHub release.

    Args:
        output_dir: Directory to save model files
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Define file paths
        model_file = "kokoro-v1_0.pth"
        config_file = "config.json"
        model_path = os.path.join(output_dir, model_file)
        config_path = os.path.join(output_dir, config_file)

        # Check if files already exist and are valid
        if verify_files(model_path, config_path):
            logger.info("Model files already exist and are valid")
            return

        logger.info("Downloading Kokoro v1.0 model files")

        # GitHub release URLs (updated if/when new model weights released)
        base_url = "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.4"
        model_url = f"{base_url}/{model_file}"
        config_url = f"{base_url}/{config_file}"

        # download to temp paths, move into place only after checksums pass
        model_tmp = model_path + ".download"
        config_tmp = config_path + ".download"
        try:
            logger.info("Downloading model file...")
            urlretrieve(model_url, model_tmp)

            logger.info("Downloading config file...")
            urlretrieve(config_url, config_tmp)

            for name, tmp in ((model_file, model_tmp), (config_file, config_tmp)):
                got = _sha256(tmp)
                if got != EXPECTED_SHA256[name]:
                    raise RuntimeError(
                        f"checksum mismatch for {name}: expected {EXPECTED_SHA256[name]}, got {got}"
                    )

            os.replace(model_tmp, model_path)
            os.replace(config_tmp, config_path)
        finally:
            for tmp in (model_tmp, config_tmp):
                if os.path.exists(tmp):
                    os.remove(tmp)

        logger.info(f"✓ Model files prepared in {output_dir}")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Kokoro v1.0 model")
    parser.add_argument(
        "--output", required=True, help="Output directory for model files"
    )

    args = parser.parse_args()
    download_model(args.output)


if __name__ == "__main__":
    main()
