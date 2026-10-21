# Real-Time Toxic Gas Leak Dispersion Simulator

[![CI](https://github.com/swastigupta8/gas-leak-pcap/actions/workflows/ci.yml/badge.svg)](https://github.com/swastigupta8/gas-leak-pcap/actions/workflows/ci.yml)

Parallel stencil computation for predicting where a toxic gas plume goes, fast enough to stay ahead of the gas. Built for the Parallel Computer Architecture and Programming course.

**Team:** Swasti Gupta, Arisha Jaiswal, Dhruvishi Amin, Manya Vishwanath (who did what: [Team](#team))

**Report:** [`docs/report/Gas_Leak_PCAP_IEEE_Report.docx`](docs/report/Gas_Leak_PCAP_IEEE_Report.docx) (editable Word file)

| Part | What it is |
|---|---|
| [`core2d/`](core2d) | 2D advection–diffusion solver (5-point stencil) in six versions: serial, OpenMP, MPI (2D decomposition, optional comm/compute overlap), and three CUDA kernels (naive, shared-memory tiled, thread-coarsened) |
| [`dashboard/`](dashboard) | Live web dashboard: click to place a leak, set the wind, watch the plume against AEGL exposure bands |
| [`sim3d/`](sim3d) | 3D extension: Lattice Boltzmann air flow over real terrain and buildings, replaying the 2020 Vizag styrene leak |
| [`tests/`](tests) | 45 tests: bit-exact agreement of every parallel version, real-GPU vs CPU checks, analytic accuracy, mass conservation |
| [`Makefile`](Makefile), [`scripts/`](scripts) | One-command setup, build, test, demos, benchmarks and report figures |
| [`docs/`](docs) | [The model](docs/model.md), [parallel design and performance model](docs/parallel_design.md), [revised synopsis](docs/synopsis.md) |

## Run it on a Windows laptop

Everything runs inside **WSL2** (Ubuntu inside Windows). NVIDIA supports CUDA there through the normal Windows driver, and the code builds exactly as on Linux.

**1. Install Ubuntu (once).** Open PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart when asked, then open **Ubuntu** from the Start menu and choose a Linux username and password.

**2. Check the GPU (NVIDIA only).** Keep the NVIDIA driver up to date on **Windows**. Do not install any NVIDIA driver inside Ubuntu. In the Ubuntu terminal:

```bash
nvidia-smi
```

If it lists your GPU, CUDA will be set up automatically in the next step. If it fails or you have no NVIDIA GPU, everything except the CUDA parts still works.

**3. Get the code and set up (once, 5–15 minutes).** Clone into your Linux home folder, not into `/mnt/c/...`: Windows folders are much slower from Linux.

```bash
cd ~
git clone https://github.com/swastigupta8/gas-leak-pcap.git
cd gas-leak-pcap
make setup
```

`make setup` installs compilers, Open MPI and Python packages, installs the CUDA toolkit if a GPU is visible, builds everything and runs the tests. It ends with `passed` and either `0 skipped` (GPU) or `5 skipped` (no GPU).

If the repository is private, `git clone` asks for a username and password. GitHub no longer accepts account passwords here: run `sudo apt install gh && gh auth login` first, then clone again.

**4. Run the demos.** Then open the address in your normal Windows browser:

```bash
make demo        # 2D dashboard  -> http://localhost:8000
make demo3d      # 3D Vizag      -> http://localhost:8001   (Ctrl+C stops a demo)
```

In the 2D dashboard, click the map to start a leak and use the sliders for wind and leak rate. The panel shows which backend is running (`cuda:<your GPU>` or `cpu-openmp`) and how many times faster than real time it runs.

**5. Produce the report numbers and figures.**

```bash
make report      # tests + 2D benchmarks + 3D experiments
```

Figures land in `results/` (2D) and `sim3d/results/` (3D). For the real Vizag terrain in the 3D model, run `make site` once (needs internet); without it the 3D model uses a synthetic test valley.

## All commands

| Command | What it does |
|---|---|
| `make setup` | One-time setup, then build and test |
| `make build` | Compile everything; CUDA targets too when `nvcc` is installed |
| `make test` | 45 tests; GPU tests skip automatically without a GPU |
| `make demo` / `make demo3d` | Live dashboards on ports 8000 / 8001 |
| `make bench` | 2D: every variant on 4000 × 4000, OpenMP and MPI scaling, GPU bandwidth → `results/` |
| `make experiments` | 3D: GPU-vs-CPU check, benchmarks, MPI, replay → `sim3d/results/` |
| `make site` | Download real Vizag terrain, buildings and 2020 weather for the 3D model |
| `make report` | `test` + `bench` + `experiments` |

Single runs, for experiments of your own:

```bash
./core2d/build/gas2d --variant serial --nx 4000 --ny 4000 --steps 20
./core2d/build/gas2d --variant omp    --nx 4000 --ny 4000 --steps 20
mpirun --oversubscribe -np 4 ./core2d/build/gas2d_mpi --nx 4000 --ny 4000 --steps 20 --overlap
./core2d/build/gas2d_cuda --variant shared --nx 4000 --ny 4000 --steps 500
```

Each prints one JSON line: ms per step, million cell updates per second (MLUPS), effective memory bandwidth, and the **real-time factor** (simulated seconds per wall-clock second).

## Key design points

- **Real-time factor, not iterations per second.** At 1 m cells the stability limit forces Δt ≈ 0.02 s, so 49 steps make one simulated second ([model.md §3](docs/model.md#3-stability-the-time-step-is-not-free)).
- **Memory-bound by design.** 9 flops per 8 bytes moved, so each GPU kernel is measured as a percentage of peak memory bandwidth ([parallel_design.md](docs/parallel_design.md)).
- **One stencil function everywhere.** `g2_stencil()` in `core2d/include/gas2d.h` is shared by every CPU and GPU version, with a fixed floating-point evaluation order, so the parallel versions are checked bit for bit against the serial one.
- **GPU kernels tested without a GPU.** `gas2d_emu` runs the kernels' per-thread code on the CPU with CUDA's thread, block and barrier semantics, so tiling bugs fail the tests on any machine.

## Verified so far

- On Ubuntu 24.04 with GCC 13 and Open MPI 4.1, from a clean copy: `make setup` completes, 40 tests pass and the 5 GPU tests skip. OpenMP and MPI (1–4 ranks, blocking and overlapped) are bit-identical to serial, and all three GPU kernels are bit-identical under emulation.
- Both dashboards start and stream on the CPU backend.
- All CUDA sources compile for sm_75 (checked with `ptxas`); CI builds them with `nvcc` 12.4 on every push.
- **Not yet run on a real GPU.** GPU timings and the real GPU-vs-CPU tests come from `make setup` / `make report` on an NVIDIA machine. Do not quote GPU numbers until then.

## Limitations

A teaching and benchmarking model: 2D, flat ground, uniform wind, constant turbulent diffusivity, first-order advection. Not a tool for real emergency decisions. Details in [model.md §5](docs/model.md#5-what-the-model-leaves-out).

## Team

| Member | Responsibility |
|---|---|
| Swasti Gupta | CUDA kernels (naive, shared-memory tiled, thread-coarsened), the CPU emulator of the CUDA execution model, GPU benchmarking, roofline performance model, report |
| Arisha Jaiswal | Mathematical model and stability analysis, serial C baseline, OpenMP parallelisation, validation against the analytic solution |
| Dhruvishi Amin | The entire 3D Visakhapatnam simulator in [`sim3d/`](sim3d), conceived, designed and built by her: D3Q19 Lattice Boltzmann air flow with Smagorinsky turbulence and buoyancy, gas transport, the real terrain/building/weather data pipeline, the single-source CPU/GPU build, its MPI version, the 2020 replay and the 3D dashboard |
| Manya Vishwanath | MPI 2D decomposition with derived datatypes and overlapped halo exchange, the equivalence test suite, the live 2D dashboard, build system and CI |
