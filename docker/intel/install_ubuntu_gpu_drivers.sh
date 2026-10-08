#!/bin/bash
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e

if [ -z "$INSTALL_DRIVER_VERSION" ]; then
    echo "Error: INSTALL_DRIVER_VERSION cannot be empty."
    exit 1
fi

apt-get update && apt-get install -y libnuma1 ocl-icd-libopencl1 --no-install-recommends && rm -rf /var/lib/apt/lists/* && \
case $INSTALL_DRIVER_VERSION in \
"24.26.30049") \
        mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.26.30049.6/intel-level-zero-gpu_1.3.30049.6_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.26.30049.6/intel-opencl-icd_24.26.30049.6_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.26.30049.6/libigdgmm12_22.3.20_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17193.4/intel-igc-core_1.0.17193.4_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17193.4/intel-igc-opencl_1.0.17193.4_amd64.deb ; \
        dpkg -i *.deb && rm -Rf /tmp/gpu_deps ; \
;; \
"24.39.31294") \
        mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/intel-level-zero-gpu_1.6.31294.12_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/intel-opencl-icd_24.39.31294.12_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/libigdgmm12_22.5.2_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17791.9/intel-igc-core_1.0.17791.9_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17791.9/intel-igc-opencl_1.0.17791.9_amd64.deb ; \
        dpkg -i *.deb && rm -Rf /tmp/gpu_deps ; \
;; \
"24.52.32224") \
        mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb ; \
        curl -L -O https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb ; \
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb ; \
        dpkg -i *.deb && rm -Rf /tmp/gpu_deps ; \
;; \
"25.31.34666") \
	mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.31.34666.3/libze-intel-gpu1_25.31.34666.3-0_amd64.deb; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.31.34666.3/intel-opencl-icd_25.31.34666.3-0_amd64.deb; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.31.34666.3/libigdgmm12_22.8.1_amd64.deb; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.31.34666.3/intel-ocloc_25.31.34666.3-0_amd64.deb; \
	curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/v2.16.0/intel-igc-core-2_2.16.0+19683_amd64.deb; \
	curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/v2.16.0/intel-igc-opencl-2_2.16.0+19683_amd64.deb; \
	dpkg -i *.deb && rm -Rf /tmp/gpu_deps ; \
;; \
"25.35.35096") \
	mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/libze-intel-gpu1_25.35.35096.9-0_amd64.deb; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-opencl-icd_25.35.35096.9-0_amd64.deb; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/libigdgmm12_22.8.1_amd64.deb; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.35.35096.9/intel-ocloc_25.35.35096.9-0_amd64.deb; \
	curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/v2.18.5/intel-igc-core-2_2.18.5+19820_amd64.deb; \
	curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/v2.18.5/intel-igc-opencl-2_2.18.5+19820_amd64.deb; \
	dpkg -i *.deb && rm -Rf /tmp/gpu_deps ; \
;; \
"25.48.36300") \
	mkdir /tmp/gpu_deps && cd /tmp/gpu_deps ; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/libze-intel-gpu1_25.48.36300.8-0_amd64.deb; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/intel-opencl-icd_25.48.36300.8-0_amd64.deb; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/libigdgmm12_22.8.2_amd64.deb; \
	curl -L -O https://github.com/intel/compute-runtime/releases/download/25.48.36300.8/intel-ocloc_25.48.36300.8-0_amd64.deb; \
	curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/v2.24.8/intel-igc-core-2_2.24.8+20344_amd64.deb; \
	curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/v2.24.8/intel-igc-opencl-2_2.24.8+20344_amd64.deb; \
	dpkg -i *.deb && rm -Rf /tmp/gpu_deps ; \
;; \
*) \
        dpkg -P intel-gmmlib intel-igc-core intel-igc-opencl intel-level-zero-gpu intel-ocloc intel-opencl intel-opencl-icd && \
        apt-get update && apt-get -y --no-install-recommends install dpkg-dev && rm -rf /var/lib/apt/lists/* && \
        cd /drivers/${INSTALL_DRIVER_VERSION} && \
            dpkg-scanpackages .  > Packages && \
            cd - ; \
        echo "deb [trusted=yes arch=amd64] file:/drivers/${INSTALL_DRIVER_VERSION} ./" > /etc/apt/sources.list.d/intel-graphics-${INSTALL_DRIVER_VERSION}.list ; \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            intel-opencl-icd \
            intel-level-zero-gpu level-zero \
            intel-media-va-driver-non-free libmfx1 && \
            rm -rf /var/lib/apt/lists/* ; \
esac

apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
