#!/usr/bin/env python3
"""Download and prepare Kokoro v1.0 model."""

import hashlib
import json
import os
import socket
import sys
from urllib.request import urlretrieve

BASE_URL = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/41e5892b9d8b43e56fc560f892312a328a410973"
FALLBACK_URL = "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.4"
KOKORO_HUB_CONFIG = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json"
TUNER_HUB_CONFIG = (
    "https://huggingface.co/remsky/kokoro-inno-clone-tuner/resolve/main/config.json"
)
# both URLs serve the same files, update all three together
EXPECTED_SHA256 = {
    "kokoro-v1_0.pth": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
    "config.json": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
}
HUB_CONFIG_SHA256 = {
    KOKORO_HUB_CONFIG: EXPECTED_SHA256["config.json"],
    TUNER_HUB_CONFIG: "319569366235a0c7fed8c34188edb6b713b53f1d56a1334196e8170800f873c8",
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class _Logger:
    def info(self, msg: str) -> None:
        _log(msg)

    def error(self, msg: str) -> None:
        _log(f"ERROR: {msg}")


logger = _Logger()


def _verify_sha256(name: str, path: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != EXPECTED_SHA256[name]:
        raise RuntimeError(
            f"checksum mismatch for {name}: expected {EXPECTED_SHA256[name]}, got {got}"
        )


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
    """Download model files from the Hub at the pinned commit, else the GitHub release.

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

        # download to temp paths, move into place only after checksums pass
        model_tmp = model_path + ".download"
        config_tmp = config_path + ".download"
        for base_url in (BASE_URL, FALLBACK_URL):
            try:
                logger.info(f"Downloading model file from {base_url}...")
                urlretrieve(f"{base_url}/{model_file}", model_tmp)

                logger.info("Downloading config file...")
                urlretrieve(f"{base_url}/{config_file}", config_tmp)

                _verify_sha256(model_file, model_tmp)
                _verify_sha256(config_file, config_tmp)

                os.replace(model_tmp, model_path)
                os.replace(config_tmp, config_path)
                break
            except Exception as e:
                if base_url == FALLBACK_URL:
                    raise
                logger.error(f"Download failed ({e}), trying {FALLBACK_URL}")
            finally:
                for tmp in (model_tmp, config_tmp):
                    if os.path.exists(tmp):
                        os.remove(tmp)

        logger.info(f"✓ Model files prepared in {output_dir}")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def download_tuner(output_dir: str) -> None:
    """Fetch the pinned Inno clone tuner weights into output_dir/inno_tuner, no-op if present."""
    dest = os.path.join(output_dir, "inno_tuner")
    path = os.path.join(dest, "model.safetensors")
    if os.path.exists(path):
        logger.info(f"Inno tuner weights already exist at {path}")
        return

    from inno_kokoro.enroll import fetch_weights

    path = fetch_weights(dest)
    logger.info(f"✓ Inno tuner weights prepared at {path}")


def _hub_get(url: str) -> bytes:
    from huggingface_hub.utils import get_session

    r = get_session().get(url, timeout=3)
    r.raise_for_status()
    return r.content


def check_hub() -> None:
    """One GET of each Hub config.json: counts a download, logs if it differs from the
    pinned digest. Never raises, skipped under HF_HUB_OFFLINE or HF_HUB_DISABLE_TELEMETRY."""
    from huggingface_hub import constants

    if constants.HF_HUB_OFFLINE or constants.HF_HUB_DISABLE_TELEMETRY:
        return
    for url, expected in HUB_CONFIG_SHA256.items():
        try:
            if hashlib.sha256(_hub_get(url)).hexdigest() != expected:
                logger.info(
                    f"{url} differs from pinned digest (changed or proxy mismatch)"
                )
        except Exception:
            pass


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Kokoro v1.0 model")
    parser.add_argument(
        "--output", required=True, help="Output directory for model files"
    )

    args = parser.parse_args()
    download_model(args.output)
    try:
        download_tuner(args.output)
    except Exception as e:
        logger.error(f"Inno tuner weights not fetched, /dev/tune will answer 503: {e}")
    check_hub()


if __name__ == "__main__":
    socket.setdefaulttimeout(30)
    main()
