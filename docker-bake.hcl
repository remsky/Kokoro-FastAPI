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

# Common settings shared between targets
target "_common" {
    context = "."
    args = {
        DEBIAN_FRONTEND = "noninteractive"
        DOWNLOAD_MODEL = "${DOWNLOAD_MODEL}"
    }
}

# Base settings for CPU builds
target "_cpu_base" {
    inherits = ["_common"]
    dockerfile = "docker/cpu/Dockerfile.optimized"
}

# Base settings for GPU builds
target "_gpu_base" {
    inherits = ["_common"]
    dockerfile = "docker/gpu/Dockerfile.optimized"
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
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}-amd64"
    ]
}

target "gpu-arm64" {
    inherits = ["_gpu_base"]
    platforms = ["linux/arm64"]
    args = {
        CUDA_VERSION = "12.9.1"
    }
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}-arm64"
    ]
}

# Blackwell / RTX 50-series variant: cu128 torch wheels (sm_120 kernels).
# x86_64 only; published as a -cu128 suffixed tag on the existing -gpu package.
target "gpu-cu128-amd64" {
    inherits = ["_gpu_base"]
    platforms = ["linux/amd64"]
    args = {
        CUDA_VERSION = "12.9.1"
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
        CUDA_VERSION = "12.9.1"
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

group "all" {
    targets = ["cpu", "gpu-amd64", "gpu-arm64", "gpu-cu128-amd64", "rocm-amd64"]
}

group "individual-platforms" {
    targets = ["cpu-amd64", "cpu-arm64", "gpu-amd64", "gpu-arm64", "gpu-cu128-amd64", "rocm-amd64"]
}
