#!/bin/bash
set -e

# Get version from argument or use default
VERSION=${1:-"latest"}

# Build images using docker buildx bake
echo "Building images..."
VERSION=$VERSION docker buildx bake --push

echo "Build complete!"
echo "Created images with version: $VERSION"
