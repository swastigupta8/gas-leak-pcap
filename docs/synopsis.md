# Real-Time Toxic Gas Leak Dispersion Simulator via Parallel Stencil Computations

**Team:** Swasti Gupta, Arisha Jaiswal, Dhruvishi Amin, Manya Vishwanath
**Course:** Parallel Computer Architecture and Programming

*Revised synopsis. Changes from the first draft are listed at the end.*

## 1. Objective

Build a toxic gas dispersion simulator that runs many times faster than real time, so emergency responders can see where a plume will be before it gets there. The core solves the 2D advection–diffusion equation with a 5-point stencil and is parallelised four ways: OpenMP, MPI, and CUDA (naive and shared-memory kernels, plus a thread-coarsened kernel). A 3D extension models the 2020 Vizag styrene leak over real terrain and buildings.

## 2. Problem and end user

**End users:** HazMat response teams, firefighters, municipal emergency planners.

**Problem:** after a chemical release, responders need to know which areas will exceed safe exposure levels, and when. A single-core CPU solves a 4000 × 4000 grid at 1 m resolution only around real time or slower. A forecast that arrives after the gas does is useless.

## 3. Mathematical model

Concentration $C$ (mg/m³), uniform wind $(u, v)$, eddy diffusivity $D$, leak $S$:

$$
\frac{\partial C}{\partial t} + u\frac{\partial C}{\partial x} + v\frac{\partial C}{\partial y} = D\nabla^2 C + S
$$

With forward-Euler time stepping, central differences for diffusion and upwind differences for advection, each cell's update needs only its four neighbours:

$$
C^{n+1}_{i,j} = w_0 C_{i,j} + w_E C_{i+1,j} + w_W C_{i-1,j} + w_N C_{i,j+1} + w_S C_{i,j-1}
$$

The weights depend on $\Delta t, h, D, u, v$ (derivation in `docs/model.md`). The scheme is stable and never produces negative concentrations when

$$
\Delta t \le \frac{1}{4D/h^2 + (|u| + |v|)/h}
$$

which at $h = 1$ m, $D = 10$ m²/s and a 4 m/s wind means $\Delta t \approx 0.02$ s: **about 49 steps per simulated second**.

## 4. Technical architecture and parallel justification

A 4000 × 4000 grid has 16 million cells. Each update does 9 floating-point operations but moves at least 8 bytes, so the computation is **memory-bandwidth bound** (arithmetic intensity ≈ 1.1 flop/byte). Every design decision targets memory traffic.

- **CUDA:** one thread per cell (1 m²), 32 × 8 thread blocks for coalesced memory access.
- **Shared memory:** each block stages its tile and a one-cell halo in on-chip shared memory, so each value is read from global memory once per block instead of up to five times. This reduces global memory traffic; it does not change the memory-bound nature of the problem.
- **Thread coarsening:** each thread updates a column of cells, reusing neighbour values held in registers.
- **OpenMP:** row blocks across CPU cores, the multi-core CPU baseline.
- **MPI:** 2D block decomposition with halo exchange using derived datatypes, plus a variant that overlaps communication with computation.

## 5. Frontend and deliverable

A web dashboard showing the map as a heat map:

- **Controls:** click to place the leak; sliders for wind speed, wind direction, leak rate and playback speed.
- **Output:** colour bands at the US EPA AEGL-1/2/3 thresholds (green below AEGL-1; yellow, orange, red above), the area inside each band, and live performance figures. Updates about 30 times per second; each frame advances many simulation steps.

## 6. Evaluation metrics

1. **Real-time factor** (primary): simulated seconds per wall-clock second on the 4000 × 4000 grid. Target ≥ 30×, i.e. a 30-minute forecast in under a minute (about 1,500 steps/s).
2. **Throughput and efficiency:** cell updates per second, and achieved memory bandwidth as a percentage of the GPU's peak.
3. **Speed-up and scaling:** each variant against the serial C baseline; OpenMP strong scaling; MPI compute vs communication time.
4. **Correctness:** parallel versions bit-identical to serial (GPU within floating-point tolerance); accuracy against an analytic solution.

## Changes from the first draft

| First draft | Revised | Why |
|---|---|---|
| Update equation missing | Equation and stability condition given | The equation is the core of the project |
| "30 iterations per second" on 4000 × 4000 | Real-time factor ≥ 30× | At 1 m cells, 30 iterations/s is 0.6× real time, slower than the gas. Iterations per second means nothing without the time step. |
| Shared memory "removes the von Neumann bottleneck" | Shared memory reduces global memory traffic | No software technique removes the processor–memory bottleneck; the stencil stays memory-bound. Claiming otherwise invites an easy examiner question. |
| Speed-up against "a C++ version" | Against a serial C baseline, plus OpenMP | The baseline must be identified exactly, and a multi-core CPU comparison is fairer to the CPU |
| Grid 2000 × 2000 in section 4, 4000 × 4000 in section 6 | 4000 × 4000 throughout; dashboard defaults to 2000 × 2000 for responsiveness | Consistency |
| "Accurate predictions" | Teaching model with stated limits; 3D extension for terrain and buildings | A 2D, flat, uniform-wind model cannot claim operational accuracy |
