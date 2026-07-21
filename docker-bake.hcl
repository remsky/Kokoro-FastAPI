# Variables for reuse
variable "VERSION" {
    default = "latest"
}

variable "REGISTRY" {
    default = "ghcr.io"
}

variable "OWNER" {
    default = "remsky"
}

variable "REPO" {
    default = "kokoro-fastapi"
}

variable "DOWNLOAD_MODEL" {
    default = "true"
}

# Source-control revision + build timestamp, populated from CI env.
# Left blank for local builds so the resulting labels/annotations stay empty
# rather than carrying stale values.
variable "REVISION" {
    default = ""
}

variable "CREATED" {
    default = ""
}

# OCI metadata applied to every image. `labels` lands in the image config
# (visible via `docker inspect`); `annotations` lands on the pushed manifest
# (which is what GHCR reads for per-arch package pages). Index-level
# annotations for the multi-arch tag are added in release.yml at
# `imagetools create` time, since bake here only produces per-arch manifests.
target "_common" {
    context = "."
    args = {
        DEBIAN_FRONTEND = "noninteractive"
        DOWNLOAD_MODEL = "${DOWNLOAD_MODEL}"
    }
    labels = {
        "org.opencontainers.image.source"   = "https://github.com/${OWNER}/Kokoro-FastAPI"
        "org.opencontainers.image.url"      = "https://github.com/${OWNER}/Kokoro-FastAPI"
        "org.opencontainers.image.licenses" = "Apache-2.0"
        "org.opencontainers.image.revision" = "${REVISION}"
        "org.opencontainers.image.version"  = "${VERSION}"
        "org.opencontainers.image.created"  = "${CREATED}"
    }
    annotations = [
        "org.opencontainers.image.source=https://github.com/${OWNER}/Kokoro-FastAPI",
        "org.opencontainers.image.url=https://github.com/${OWNER}/Kokoro-FastAPI",
        "org.opencontainers.image.licenses=Apache-2.0",
        "org.opencontainers.image.revision=${REVISION}",
        "org.opencontainers.image.version=${VERSION}",
        "org.opencontainers.image.created=${CREATED}",
    ]
}

# Base settings for CPU builds
target "_cpu_base" {
    inherits = ["_common"]
    dockerfile = "docker/cpu/Dockerfile.optimized"
    labels = {
        "org.opencontainers.image.title"       = "Kokoro-FastAPI (CPU)"
        "org.opencontainers.image.description" = "Kokoro TTS served via FastAPI. CPU build."
    }
    annotations = [
        "org.opencontainers.image.title=Kokoro-FastAPI (CPU)",
        "org.opencontainers.image.description=Kokoro TTS served via FastAPI. CPU build.",
    ]
}

# Base settings for GPU builds
target "_gpu_base" {
    inherits = ["_common"]
    dockerfile = "docker/gpu/Dockerfile.optimized"
    labels = {
        "org.opencontainers.image.title"       = "Kokoro-FastAPI (GPU)"
        "org.opencontainers.image.description" = "Kokoro TTS served via FastAPI. NVIDIA GPU build (CUDA 12.6 amd64 / CUDA 12.9 arm64; cu128 tag for Blackwell)."
    }
    annotations = [
        "org.opencontainers.image.title=Kokoro-FastAPI (GPU)",
        "org.opencontainers.image.description=Kokoro TTS served via FastAPI. NVIDIA GPU build (CUDA 12.6 amd64 / CUDA 12.9 arm64; cu128 tag for Blackwell).",
    ]
}

# CPU target with multi-platform support
target "cpu" {
    inherits = ["_cpu_base"]
    platforms = ["linux/amd64", "linux/arm64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}"
    ]
}

# GPU multi-platform: dispatches to per-arch targets so each gets its own CUDA_VERSION
group "gpu" {
    targets = ["gpu-amd64", "gpu-arm64"]
}

# Base settings for AMD ROCm builds
target "_rocm_base" {
    inherits = ["_common"]
    dockerfile = "docker/rocm/Dockerfile"
    labels = {
        "org.opencontainers.image.title"       = "Kokoro-FastAPI (ROCm)"
        "org.opencontainers.image.description" = "Kokoro TTS served via FastAPI. AMD ROCm build (amd64 only)."
    }
    annotations = [
        "org.opencontainers.image.title=Kokoro-FastAPI (ROCm)",
        "org.opencontainers.image.description=Kokoro TTS served via FastAPI. AMD ROCm build (amd64 only).",
    ]
}

# Base settings for Intel builds
target "_intel_base" {
    inherits = ["_common"]
    dockerfile = "docker/intel/Dockerfile"
    labels = {
        "org.opencontainers.image.title"       = "Kokoro-FastAPI (Intel)"
        "org.opencontainers.image.description" = "Kokoro TTS served via FastAPI. Intel GPU build (amd64 only)."
    }
    annotations = [
        "org.opencontainers.image.title=Kokoro-FastAPI (Intel)",
        "org.opencontainers.image.description=Kokoro TTS served via FastAPI. Intel GPU build (amd64 only).",
    ]
}



# Individual platform targets for debugging/testing
target "cpu-amd64" {
    inherits = ["_cpu_base"]
    platforms = ["linux/amd64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}-amd64"
    ]
}

target "cpu-arm64" {
    inherits = ["_cpu_base"]
    platforms = ["linux/arm64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}-arm64"
    ]
}

target "gpu-amd64" {
    inherits = ["_gpu_base"]
    platforms = ["linux/amd64"]
    args = {
        CUDA_VERSION = "12.6.3"
    }
    # Per-arch tag carries the wheel variant so it parallels gpu-cu128-amd64.
    # The published manifest still resolves to :VERSION / :VERSION-cu126 via release.yml.
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}-cu126-amd64"
    ]
}

target "gpu-arm64" {
    inherits = ["_gpu_base"]
    platforms = ["linux/arm64"]
    args = {
        CUDA_VERSION = "12.9.1"
    }
    # aarch64 uses cu129 wheels (no cu126 aarch64 wheels exist on pytorch.org).
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}-cu129-arm64"
    ]
}

# Blackwell / RTX 50-series variant: cu128 torch wheels (sm_120 kernels).
# x86_64 only; published as a -cu128 suffixed tag on the existing -gpu package.
target "gpu-cu128-amd64" {
    inherits = ["_gpu_base"]
    platforms = ["linux/amd64"]
    args = {
        # 12.8.x is the first CUDA toolkit with Blackwell (sm_120) support and is
        # what the cu128 torch wheels are built against. Keep base + wheel aligned.
        CUDA_VERSION = "12.8.1"
        GPU_EXTRA = "gpu-cu128"
    }
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}-cu128-amd64"
    ]
}

# AMD ROCm only supports x86
target "rocm-amd64" {
    inherits = ["_rocm_base"]
    platforms = ["linux/amd64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-rocm:${VERSION}-amd64"
    ]
}

# Intel only supports x86
target "intel-amd64" {
    inherits = ["_intel_base"]
    platforms = ["linux/amd64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-intel:${VERSION}-amd64"
    ]
}

# Development targets for faster local builds
target "cpu-dev" {
    inherits = ["_cpu_base"]
    # No multi-platform for dev builds
    tags = ["${REGISTRY}/${OWNER}/${REPO}-cpu:dev"]
}

target "gpu-dev" {
    inherits = ["_gpu_base"]
    # No multi-platform for dev builds
    tags = ["${REGISTRY}/${OWNER}/${REPO}-gpu:dev"]
}

target "gpu-cu128-dev" {
    inherits = ["_gpu_base"]
    # No multi-platform for dev builds
    args = {
        CUDA_VERSION = "12.8.1"
        GPU_EXTRA = "gpu-cu128"
    }
    tags = ["${REGISTRY}/${OWNER}/${REPO}-gpu:dev-cu128"]
}

group "dev" {
    targets = ["cpu-dev", "gpu-dev"]
}

# Build groups for different use cases
group "cpu-all" {
    targets = ["cpu", "cpu-amd64", "cpu-arm64"]
}

group "gpu-all" {
    targets = ["gpu-amd64", "gpu-arm64", "gpu-cu128-amd64"]
}

group "rocm-all" {
    targets = ["rocm-amd64"]
}

group "intel-all" {
    targets = ["intel-amd64"]
}

group "all" {
    targets = ["cpu", "gpu-amd64", "gpu-arm64", "gpu-cu128-amd64", "rocm-amd64", "intel-amd64"]
}

group "individual-platforms" {
    targets = ["cpu-amd64", "cpu-arm64", "gpu-amd64", "gpu-arm64", "gpu-cu128-amd64", "rocm-amd64", "intel-amd64"]
}
