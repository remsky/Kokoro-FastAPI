#!/bin/bash

set -e

ver() {
    printf "%3d%03d%03d%03d" $(echo "$1" | tr '.' ' ');
}

# Sets GFX_ARCH to default if not set
if [ -z "$GFX_ARCH" ]; then
    echo "WARNING: missing env var GFX_ARCH, using default (this will take longer)"
    GFX_ARCHS=("gfx900" "gfx906" "gfx908" "gfx90a" "gfx942" "gfx1030")
else
    # Convert ; seperated string to array
    IFS=';' read -ra GFX_ARCHS <<< "$GFX_ARCH"
fi

# Sets ROCM_VERSION to "latest" if not set
if [ -z "$ROCM_VERSION" ]; then
    echo "WARNING: missing env var ROCM_VERSION, using latest kdb repo (NOT RECOMMENDED)"
    ROCM_VERSION="latest"
fi

# Set PyTorch version and wheel install path
TORCH_INSTALL_PATH=$(uv pip show torch | grep Location | cut -d" " -f 2)

# Check if Torch installation path exists
if [ ! -d "$TORCH_INSTALL_PATH" ]; then
    echo "Error: Torch installation path '$TORCH_INSTALL_PATH' does not exist."
    exit 1
fi

# Print variable overview
echo "ROCM version: $ROCM_VERSION"
echo "GFX architectures: ${GFX_ARCHS[@]}"
echo "PyTorch installation path: $TORCH_INSTALL_PATH"

# Create directory for extraction
EXTRACT_DIR=extract_miopen_dbs
rm -rf $EXTRACT_DIR
mkdir -p "$EXTRACT_DIR" && cd "$EXTRACT_DIR"

if [[ -f /etc/lsb-release ]]; then
    # Exit if not 20.04, 22.04, or 24.04
    source /etc/lsb-release
    echo "DISTRIB_RELEASE: $DISTRIB_RELEASE"
    if [[ "$DISTRIB_RELEASE" != "20.04" && "$DISTRIB_RELEASE" != "22.04" ]]; then
        if [[ "$ROCM_VERSION" != "latest" && $(ver $ROCM_VERSION) -lt $(ver 6.2) && "$DISTRIB_RELEASE" == "24.04" ]]; then
             echo "ERROR: Unsupported Ubuntu version."
             exit 1
        fi
    fi

    for arch in "${GFX_ARCHS[@]}"; do
        # Download MIOpen .kdbs for ROCm version and GPU architecture on ubuntu
        echo "Downloading .kdb files for rocm-$ROCM_VERSION ($arch arch) ..."
        wget -q -r -np -nd -A miopen-hip-$arch*kdb_*$DISTRIB_RELEASE*deb \
            https://repo.radeon.com/rocm/apt/$ROCM_VERSION/pool/main/m/

        # Check if files were downloaded. No KDB files in repo.radeon will result in error.
        if ! ls miopen-hip-$arch*kdb_*$DISTRIB_RELEASE*deb 1> /dev/null 2>&1; then
            echo -e "ERROR: No MIOpen kernel database files found for $arch\nPlease check https://repo.radeon.com/rocm/apt/$ROCM_VERSION/pool/main/m/ for supported architectures"
            exit 1
        fi
    done

    # Extract all .deb files to local directory
    echo "Extracting deb packages for ${GFX_ARCHS[@]} ..."
    for deb_file in `ls *deb`; do
        echo "Extracting $deb_file..."
        dpkg-deb -xv "$deb_file" . > /dev/null 2>&1
    done

elif [[ -f /etc/centos-release || -f /etc/redhat-release ]]; then
    # Centos kdbs
    source /etc/os-release && RHEL_VERSION="$VERSION_ID"
    RHEL_MAJOR_VERSION=${RHEL_VERSION%%.*}
    echo "RHEL_VERSION: $RHEL_VERSION; RHEL_MAJOR_VERSION: $RHEL_MAJOR_VERSION"
    if [[ ! "$RHEL_VERSION" =~ ^(8|9) ]]; then
        echo "ERROR: Unsupported CentOS/RHEL release"
    fi
    for arch in "${GFX_ARCHS[@]}"; do
        # Download MIOpen .kdbs for ROCm version and GPU architecture on centos
        echo "Downloading .kdb files for rocm-$ROCM_VERSION ($arch arch) ..."
        wget -q -r -np -nd -A miopen-hip-$arch*kdb-[0-9]*rpm \
            https://repo.radeon.com/rocm/rhel${RHEL_MAJOR_VERSION}/$ROCM_VERSION/main

        # Check if files were downloaded. No KDB files in repo.radeon will result in error.
        if ! ls miopen-hip-$arch*kdb-*rpm 1> /dev/null 2>&1; then
            echo -e "ERROR: No MIOpen kernel database files found for $arch\nPlease check https://repo.radeon.com/rocm/rhel${RHEL_MAJOR_VERSION}/$ROCM_VERSION/main for supported architectures"
            exit 1
        fi
    done

    # Extract all RPM files to current directory
    echo "Extracting rpm packages for ${GFX_ARCHS[@]} ..."
    for rpm_file in `ls *rpm`; do
        echo "Extracting $rpm_file..."
        rpm2cpio "$rpm_file" | cpio -idmv 2> /dev/null
    done
else
    echo "ERROR: Unsupported operating system."
    exit 1
fi

# Copy miopen db files to PyTorch installation path
echo "Copying kdb files to ${TORCH_INSTALL_PATH}/torch/share"
cp -ra opt/rocm-*/share/miopen $TORCH_INSTALL_PATH/torch/share

# Remove downloaded files and extract directory
cd .. && rm -rf $EXTRACT_DIR
echo "Successfully installed MIOpen kernel database files"
