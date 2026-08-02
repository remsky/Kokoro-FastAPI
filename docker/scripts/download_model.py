#!/usr/bin/env python3
"""Download and prepare Kokoro model weights from GitHub release assets."""

import hashlib
import os
import sys
import tarfile
from urllib.request import urlretrieve

# release that hosts the weight assets, update alongside the sha256 tables
BASE_URL = "https://github.com/remsky/Kokoro-FastAPI/releases/download/v0.1.4"

# per-version model assets: local name -> (release asset name, sha256)
MODEL_FILES = {
    "v1_0": {
        "kokoro-v1_0.pth": (
            "kokoro-v1_0.pth",
            "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
        ),
        "config.json": (
            "config.json",
            "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
        ),
    },
    "v1_1-zh": {
        "kokoro-v1_1-zh.pth": (
            "kokoro-v1_1-zh.pth",
            "b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b",
        ),
        "config.json": (
            "config-v1_1-zh.json",
            "bc333efa5ce4ceff433c8c8e5d027a1eca0166001e4e4a62bea2d26ff7a46890",
        ),
    },
}

# voice packs downloaded as tarballs: (asset name, sha256, voice count); v1_0 voices ship in the repo
VOICE_PACKS = {
    "v1_1-zh": (
        "voices-v1_1-zh.tar.gz",
        "875acc33452139582bdac8b6d6a4e490127a4f20bb5785432d5581e12298c238",
        103,
    ),
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


def _fetch_verified(asset: str, expected_sha256: str, dest: str) -> None:
    """Download a release asset to dest, checksum-verified via a temp path."""
    # pid-scoped so concurrent starts on a shared volume don't collide
    tmp = f"{dest}.{os.getpid()}.download"
    try:
        urlretrieve(f"{BASE_URL}/{asset}", tmp)
        got = _sha256(tmp)
        if got != expected_sha256:
            raise RuntimeError(
                f"checksum mismatch for {asset}: expected {expected_sha256}, got {got}"
            )
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def verify_files(output_dir: str, version: str) -> bool:
    """Check on-disk model files against the pinned checksums.

    Args:
        output_dir: Directory holding the model files
        version: Model version key in MODEL_FILES

    Returns:
        True if every file is present and matches its sha256
    """
    # checksums also reject truncated files and error-page bodies (#301)
    try:
        for local_name, (_asset, sha256) in MODEL_FILES[version].items():
            path = os.path.join(output_dir, local_name)
            if not os.path.exists(path) or _sha256(path) != sha256:
                return False
        return True
    except Exception:
        return False


def download_voices(version: str, voices_dir: str) -> None:
    """Download and extract the voice pack for versions not shipped in the repo.

    Args:
        version: Model version key in VOICE_PACKS
        voices_dir: Directory to extract voice .pt files into
    """
    if version not in VOICE_PACKS:
        return
    asset, sha256, count = VOICE_PACKS[version]

    os.makedirs(voices_dir, exist_ok=True)
    have = [f for f in os.listdir(voices_dir) if f.endswith(".pt")]
    if len(have) >= count:
        logger.info(f"Voice pack already present in {voices_dir}")
        return

    logger.info(f"Downloading voice pack {asset}...")
    tarball = os.path.join(voices_dir, asset)
    try:
        _fetch_verified(asset, sha256, tarball)
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(voices_dir, filter="data")
    finally:
        if os.path.exists(tarball):
            os.remove(tarball)
    logger.info(f"✓ Voice pack extracted to {voices_dir}")


def download_model(output_dir: str, version: str = "v1_0") -> None:
    """Download model files from GitHub release.

    Args:
        output_dir: Directory to save model files
        version: Model version to download (key in MODEL_FILES)
    """
    try:
        if version not in MODEL_FILES:
            raise ValueError(
                f"Unknown model version '{version}', expected one of {sorted(MODEL_FILES)}"
            )
        files = MODEL_FILES[version]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Check if files already exist and are valid
        if verify_files(output_dir, version):
            logger.info("Model files already exist and are valid")
            return

        logger.info(f"Downloading Kokoro {version} model files")
        for local_name, (asset, sha256) in files.items():
            logger.info(f"Downloading {asset}...")
            _fetch_verified(asset, sha256, os.path.join(output_dir, local_name))

        logger.info(f"✓ Model files prepared in {output_dir}")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Kokoro model weights")
    parser.add_argument(
        "--output", required=True, help="Output directory for model files"
    )
    parser.add_argument(
        "--version", default="v1_0", help="Model version (v1_0 or v1_1-zh)"
    )
    parser.add_argument(
        "--voices-output",
        default=None,
        help="Voices directory, for versions whose voice pack is not shipped in the repo",
    )

    args = parser.parse_args()
    download_model(args.output, args.version)
    if args.voices_output:
        download_voices(args.version, args.voices_output)


if __name__ == "__main__":
    main()
