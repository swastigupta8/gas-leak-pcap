# Makefile - every way to run the project, from the repository root.
#
#   make setup        one-time: system packages, Python env, CUDA (if a GPU is present), build, test
#   make build        compile everything (CUDA targets too, when nvcc is installed)
#   make test         full test suite (GPU tests run only when a GPU and the CUDA builds exist)
#   make demo         2D live dashboard        -> http://localhost:8000
#   make demo3d       3D Vizag dashboard       -> http://localhost:8001
#   make bench        2D benchmarks + figures  -> results/
#   make experiments  3D check, benchmarks, MPI, replay + figures -> sim3d/results/
#   make site         download the real Vizag terrain/buildings/weather for the 3D model
#   make report       test + bench + experiments: every number and figure for the report
#   make clean

# The CUDA toolkit installs to /usr/local/cuda/bin, which may not be on PATH in this terminal yet.
export PATH := /usr/local/cuda/bin:$(PATH)

# Use the project's virtual environment if `make setup` created it (absolute path, so it still
# works after `cd` into a subfolder).
PY := $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/python,python3)
# Make 4.3 (Ubuntu 22.04/24.04) does not pass the exported PATH to $(shell), so also look in the
# toolkit's default location explicitly.
NVCC := $(or $(shell command -v nvcc 2>/dev/null),$(wildcard /usr/local/cuda/bin/nvcc))

.DEFAULT_GOAL := build
.PHONY: setup build test demo demo3d bench experiments site report clean

setup:
	bash scripts/setup.sh

build:
	$(MAKE) -C core2d
	$(MAKE) -C sim3d
ifneq ($(NVCC),)
	$(MAKE) -C core2d cuda CUDA_ARCH=native
	$(MAKE) -C sim3d cuda CUDA_ARCH=native
else
	@echo "nvcc not found: built the CPU versions only."
endif

test: build
	$(PY) -m pytest tests -v -rs

demo: build
	$(PY) dashboard/server.py

demo3d: build
	cd sim3d && PORT=8001 $(PY) server/server.py

bench: build
	$(PY) scripts/benchmark.py

experiments: build
	$(PY) sim3d/experiments.py all

site:
	$(PY) sim3d/experiments.py site

report: test bench experiments
	@echo "Figures: results/ (2D) and sim3d/results/ (3D)"

clean:
	$(MAKE) -C core2d clean
	$(MAKE) -C sim3d clean
	rm -rf results sim3d/results
