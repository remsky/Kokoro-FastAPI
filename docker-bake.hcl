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
    dockerfile = "docker/cpu/Dockerfile"
}

# Base settings for GPU builds
target "_gpu_base" {
    inherits = ["_common"]
    dockerfile = "docker/gpu/Dockerfile"
}

# Base settings for AMD ROCm builds
target "_rocm_base" {
    inherits = ["_common"]
    dockerfile = "docker/rocm/Dockerfile"
}

# CPU target with multi-platform support
target "cpu" {
    inherits = ["_cpu_base"]
    platforms = ["linux/amd64", "linux/arm64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}",
        "${REGISTRY}/${OWNER}/${REPO}-cpu:latest"
    ]
}

# GPU target with multi-platform support
target "gpu" {
    inherits = ["_gpu_base"]
    platforms = ["linux/amd64", "linux/arm64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}",
        "${REGISTRY}/${OWNER}/${REPO}-gpu:latest"
    ]
}

# AMD ROCm target with multi-platform support
target "rocm" {
    inherits = ["_rocm_base"]
    platforms = ["linux/amd64", "linux/arm64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-rocm:${VERSION}",
        "${REGISTRY}/${OWNER}/${REPO}-rocm:latest"
    ]
}

# Default group to build both CPU and GPU versions
group "default" {
    targets = ["cpu", "gpu", "rocm"]
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

target "rocm-dev" {
    inherits = ["_rocm_base"]
    # No multi-platform for dev builds
    tags = ["${REGISTRY}/${OWNER}/${REPO}-rocm:dev"]
}

group "dev" {
    targets = ["cpu-dev", "gpu-dev", "rocm-dev"]
}
