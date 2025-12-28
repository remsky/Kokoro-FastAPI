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
        "${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}-amd64",
        "${REGISTRY}/${OWNER}/${REPO}-cpu:latest-amd64"
    ]
}

target "cpu-arm64" {
    inherits = ["_cpu_base"]
    platforms = ["linux/arm64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-cpu:${VERSION}-arm64",
        "${REGISTRY}/${OWNER}/${REPO}-cpu:latest-arm64"
    ]
}

target "gpu-amd64" {
    inherits = ["_gpu_base"]
    platforms = ["linux/amd64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}-amd64",
        "${REGISTRY}/${OWNER}/${REPO}-gpu:latest-amd64"
    ]
}

target "gpu-arm64" {
    inherits = ["_gpu_base"]
    platforms = ["linux/arm64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-gpu:${VERSION}-arm64",
        "${REGISTRY}/${OWNER}/${REPO}-gpu:latest-arm64"
    ]
}

# AMD ROCm only supports x86
target "rocm-amd64" {
    inherits = ["_rocm_base"]
    platforms = ["linux/amd64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}-rocm:${VERSION}-amd64",
        "${REGISTRY}/${OWNER}/${REPO}-rocm:latest-amd64"
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

group "dev" {
    targets = ["cpu-dev", "gpu-dev"]
}

# Build groups for different use cases
group "cpu-all" {
    targets = ["cpu", "cpu-amd64", "cpu-arm64"]
}

group "gpu-all" {
    targets = ["gpu", "gpu-amd64", "gpu-arm64"]
}

group "rocm-all" {
    targets = ["rocm-amd64"]
}

group "all" {
    targets = ["cpu", "gpu", "rocm"]
}

group "individual-platforms" {
    targets = ["cpu-amd64", "cpu-arm64", "gpu-amd64", "gpu-arm64", "rocm-amd64"]
}
