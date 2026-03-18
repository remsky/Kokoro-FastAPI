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

# Base settings for builds
target "_base" {
    inherits = ["_common"]
    dockerfile = "docker/Dockerfile"
}

# Default target with multi-platform support
target "default" {
    inherits = ["_base"]
    platforms = ["linux/amd64", "linux/arm64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}:${VERSION}",
        "${REGISTRY}/${OWNER}/${REPO}:latest"
    ]
}

# Individual platform targets for debugging/testing
target "amd64" {
    inherits = ["_base"]
    platforms = ["linux/amd64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}:${VERSION}-amd64",
        "${REGISTRY}/${OWNER}/${REPO}:latest-amd64"
    ]
}

target "arm64" {
    inherits = ["_base"]
    platforms = ["linux/arm64"]
    tags = [
        "${REGISTRY}/${OWNER}/${REPO}:${VERSION}-arm64",
        "${REGISTRY}/${OWNER}/${REPO}:latest-arm64"
    ]
}

# Development target for faster local builds
target "dev" {
    inherits = ["_base"]
    tags = ["${REGISTRY}/${OWNER}/${REPO}:dev"]
}

group "all" {
    targets = ["default"]
}

group "individual-platforms" {
    targets = ["amd64", "arm64"]
}
