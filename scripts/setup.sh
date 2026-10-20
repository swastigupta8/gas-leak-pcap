#!/usr/bin/env bash
# setup.sh - one-time setup on Ubuntu (WSL2 on Windows, or native Linux). Run via `make setup`.
#
#   1. system packages: compilers, Open MPI, Python
#   2. a Python virtual environment in .venv with requirements.txt
#   3. the CUDA toolkit (nvcc), only if an NVIDIA GPU is visible and nvcc is missing
#   4. builds everything and runs the test suite
#
# On WSL2: install the normal NVIDIA driver on WINDOWS only. Never install a Linux NVIDIA driver
# inside WSL; the Windows driver exposes the GPU to Linux (that is why we install
# `cuda-toolkit-12-6`, which contains no driver, and not the `cuda` meta-package, which does).
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "1/4 System packages (asks for your Linux password)"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential make openmpi-bin libopenmpi-dev python3 python3-venv python3-pip wget

step "2/4 Python environment (.venv)"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "installed: $(.venv/bin/python -c 'import numpy, fastapi, matplotlib; print("numpy", numpy.__version__, "| fastapi", fastapi.__version__)')"

step "3/4 CUDA"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    if command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/nvcc ]; then
        echo "CUDA toolkit already installed."
    else
        if grep -qi microsoft /proc/version; then
            repo="wsl-ubuntu"                                  # NVIDIA's repository for WSL2
        else
            . /etc/os-release
            repo="ubuntu${VERSION_ID//./}"                     # e.g. ubuntu2404
        fi
        echo "Installing the CUDA 12.6 toolkit from NVIDIA's $repo repository (about 3 GB)..."
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/cuda-keyring_1.1-1_all.deb" \
            -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb
        sudo apt-get update -qq
        # 12.6 supports the widest range of GPUs (CUDA 13 dropped GPUs older than Turing). If it is
        # ever withdrawn, fall back to the newest toolkit. Both are toolkit-only packages: the
        # `cuda`, `cuda-13` and `cuda-drivers` packages would pull in a Linux driver, which breaks WSL.
        sudo apt-get install -y -qq cuda-toolkit-12-6 || sudo apt-get install -y -qq cuda-toolkit
    fi
    # Make nvcc available in future terminals too (the Makefile adds it for itself either way).
    if ! grep -q '/usr/local/cuda/bin' ~/.bashrc; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    fi
else
    echo "No NVIDIA GPU visible: building the CPU versions only (the demo still works)."
    if grep -qi microsoft /proc/version; then
        echo "On WSL2: update the NVIDIA driver on Windows, then run 'nvidia-smi' in this terminal."
        echo "Once it prints your GPU, run 'make setup' again to add CUDA."
    fi
fi

step "4/4 Build and test"
make build
make test

printf '\n\033[1mDone.\033[0m Next:  make demo   (then open http://localhost:8000 in your Windows browser)\n'
