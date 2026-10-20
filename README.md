# Real-Time Toxic Gas Leak Dispersion Simulator

[![CI](https://github.com/swastigupta8/gas-leak-pcap/actions/workflows/ci.yml/badge.svg)](https://github.com/swastigupta8/gas-leak-pcap/actions/workflows/ci.yml)

Parallel stencil computation for predicting where a toxic gas plume goes, fast enough to be ahead of the gas. Built for the Parallel Computer Architecture and Programming course.

**Team:** Swasti Gupta, Arisha Jaiswal, Dhruvishi Amin, Manya Vishwanath

| Part | What it is |
|---|---|
| [`core2d/`](core2d) | The 2D advection–diffusion solver (5-point stencil) in six versions: serial, OpenMP, MPI (2D decomposition, optional comm/compute overlap), and three CUDA kernels (naive, shared-memory tiled, thread-coarsened) |
| [`dashboard/`](dashboard) | Live web dashboard: click to place a leak, set the wind, watch the plume against AEGL exposure bands |
| [`sim3d/`](sim3d) | 3D extension: Lattice Boltzmann air flow over real terrain and buildings, replaying the 2020 Vizag styrene leak |
| [`tests/`](tests) | 40 tests: bit-exact agreement of every parallel version, analytic accuracy, mass conservation |
| [`notebooks/colab_run_all.ipynb`](notebooks/colab_run_all.ipynb) | Builds and runs everything on a free Colab T4 GPU, including all benchmarks and figures |
| [`docs/`](docs) | [The model](docs/model.md), [parallel design and performance model](docs/parallel_design.md), [revised synopsis](docs/synopsis.md) |

## Quick start: Colab (GPU)

1. Open [`notebooks/colab_run_all.ipynb`](notebooks/colab_run_all.ipynb) in Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all** (about 20 minutes). The last cell prints links to both dashboards.

## Local build: Linux or WSL2

On Windows, use WSL2 (Ubuntu): the code relies on GCC's OpenMP and on a Linux MPI.

```bash
sudo apt install build-essential openmpi-bin libopenmpi-dev python3-pip
pip install numpy pytest matplotlib fastapi "uvicorn[standard]"

make -C core2d                 # CPU: gas2d, gas2d_mpi, gas2d_emu, libgas2d_cpu.so
make -C core2d cuda            # GPU (needs the CUDA toolkit; CUDA_ARCH=sm_86 for an RTX 30xx)
python -m pytest tests -q      # 40 tests
python dashboard/server.py     # http://localhost:8000 (uses the GPU library if it is built)
```

Run one experiment:

```bash
./core2d/build/gas2d --variant serial --nx 4000 --ny 4000 --steps 20
./core2d/build/gas2d --variant omp    --nx 4000 --ny 4000 --steps 20
mpirun -np 4 ./core2d/build/gas2d_mpi --nx 4000 --ny 4000 --steps 20 --overlap
./core2d/build/gas2d_cuda --variant shared --nx 4000 --ny 4000 --steps 500
```

Each prints one JSON line: ms per step, million cell updates per second (MLUPS), effective bandwidth, and the **real-time factor** (simulated seconds per wall-clock second). `python scripts/benchmark.py` runs the full set and writes figures to `results/`.

## Key design points

- **Real-time factor, not iterations per second.** At 1 m cells the stability limit forces Δt ≈ 0.02 s, so 49 steps make one simulated second. See [model.md §3](docs/model.md#3-stability-the-time-step-is-not-free).
- **Memory-bound by design.** 9 flops per 8 bytes moved. Every kernel is measured as a percentage of peak memory bandwidth; see [parallel_design.md](docs/parallel_design.md).
- **One stencil function everywhere.** `g2_stencil()` in `core2d/include/gas2d.h` is the same inline function in every CPU and GPU version, with a fixed floating-point evaluation order, so the parallel versions can be checked bit for bit against the serial one.
- **GPU kernels tested without a GPU.** `gas2d_emu` runs the kernels' per-thread code on the CPU with CUDA's thread, block and barrier semantics, so tiling bugs fail the test suite on any laptop.

## Verified so far

- All 40 tests pass on CPU (2 cores, MPICH, GCC 13): OpenMP and MPI on 1–4 ranks, blocking and overlapped, bit-identical to serial; all three GPU kernels bit-identical under emulation.
- All CUDA sources compile for sm_75 (T4), checked with `ptxas`; CI builds them with `nvcc` on every push.
- **Not yet run on a real GPU.** GPU timings and the GPU-vs-CPU comparison come from the Colab notebook. Do not quote GPU numbers until it has been run.

## Limitations

A teaching and benchmarking model: 2D, flat ground, uniform wind, constant turbulent diffusivity, first-order advection. It is not a tool for real emergency decisions. Details in [model.md §5](docs/model.md#5-what-the-model-leaves-out).
