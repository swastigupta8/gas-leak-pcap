# Parallel design and performance model

How each implementation parallelises the stencil, what limits its speed, and what to expect before measuring. Run `notebooks/colab_run_all.ipynb` on a T4 to fill in `results/`.

## 1. The bottleneck is memory, not arithmetic

Per cell and step, the update does 5 multiplies and 4 adds (9 floating-point operations). It must read at least one value and write one: 8 bytes, if neighbour re-reads are served from cache.

$$
\text{arithmetic intensity} = \frac{9 \text{ flop}}{8 \text{ B}} \approx 1.1 \text{ flop/byte}
$$

A Tesla T4 does about 8.1 TFLOP/s (FP32) against 320 GB/s of memory bandwidth. Below roughly 25 flop/byte (8100 / 320), a kernel is limited by memory. At 1.1 flop/byte, the stencil is deep in memory-bound territory: the GPU's arithmetic units will be idle about 95% of the time no matter how the code is written.

Consequences:

- The upper bound is bandwidth ÷ 8 B = **40 billion cell updates per second** on a T4.
- Each GPU kernel is judged by **achieved bandwidth as a % of peak**, which `gas2d_cuda` reports (`pct_of_peak`). Good stencil kernels reach 60–80%.
- At 60–80% of peak, a 4000 × 4000 step takes about 0.5–0.67 ms: **about 30–40× faster than real time**. *Prediction, to be confirmed by the notebook.*

The same reasoning applies to the CPU: a laptop's ~20–50 GB/s caps it at a few billion updates per second, whatever the core count.

## 2. Implementations

| Variant | File | Parallelism | Key idea |
|---|---|---|---|
| serial | `src/common.c` | none | Baseline; inner loop vectorised (SIMD) by the compiler |
| OpenMP | `src/openmp.c` | threads, shared memory | One contiguous block of rows per thread |
| MPI | `src/mpi_main.c` | processes, distributed memory | 2D block decomposition, halo exchange |
| CUDA naive | `src/cuda_backend.cu` | one GPU thread per cell | Coalesced reads, caches absorb neighbour re-reads |
| CUDA shared | same | one thread per cell, 32×8 tiles | Tile + halo staged in shared memory |
| CUDA coarse | same | one thread per 8 cells in a column | Sliding register window, 1 new vertical read per cell |

### OpenMP

Rows are split into equal contiguous blocks, one per thread. Every cell costs the same, so static partitioning is perfectly balanced and adds no scheduling overhead. Each thread reads only the input array and writes only its own rows of the output, so a step needs no locks. The implicit barrier at the end of the parallel region separates steps.

Measured pitfall (documented in `openmp.c`): wrapping the loop in `#pragma omp parallel for` made GCC lose the `restrict` no-aliasing guarantee. Vectorisation broke, and one OpenMP thread ran about 1.5× slower than serial. Each thread now calls the vectorised serial sweep on its own rows.

Expected scaling: near-linear while the grid fits in cache, then flattening once all cores share one memory bus. That is the memory-bound limit from section 1, not an OpenMP problem.

### MPI

The map is cut into a √P × √P grid of blocks (`MPI_Dims_create` + `MPI_Cart_create`). Before each step, every rank exchanges its edge cells with up to four neighbours:

- Rows are contiguous and are sent directly.
- Columns are strided (one value per row), so an `MPI_Type_vector` derived datatype describes them and no manual packing is needed.
- At the map edge the neighbour is `MPI_PROC_NULL`: the call does nothing and the ghost cells stay 0.
- `MPI_Sendrecv` pairs each send with the matching receive, so the exchange cannot deadlock.

Why blocks and not strips: on an N × N grid with P ranks, strips exchange 2N cells per rank regardless of P, while square blocks exchange about 4N/√P. The communication-to-computation ratio falls as P grows.

`--overlap` posts non-blocking `MPI_Isend`/`MPI_Irecv`, updates the inner cells (which need no halo) while messages travel, then waits and updates the one-cell ring at the block edge. `comm_ms` reports only the waiting that could not be hidden.

On a single Colab machine, all ranks share one CPU, so MPI demonstrates correctness and the compute/communication split rather than cluster-scale speed-up.

### CUDA

All three kernels launch once per step. The launch boundary is the global barrier between steps: every cell of step n must be written before any cell of step n+1 reads it. Blocks are 32 × 8 threads, so a warp covers 32 consecutive floats: one 128-byte coalesced transaction per row.

- **naive**: five global loads per thread. Neighbouring threads load overlapping cells, and the L1/texture cache on Turing serves most of those repeats.
- **shared**: the block first copies its 32 × 8 tile plus a one-cell halo (34 × 10 floats, 1,360 bytes) into shared memory. After `__syncthreads()`, it computes from the tile. No thread may return before the barrier, which is why the bounds checks live inside the two phases.
- **coarse**: each thread walks 8 cells up a column, keeping the south, centre and north values in registers. Each step north needs one new vertical read instead of three.

Compiled resource use for sm_75 (from `ptxas -v`): naive 18 registers, shared 22 registers + 1,360 B shared memory, coarse 19 registers. All are low enough for full occupancy.

**Prediction to test:** because the naive kernel's re-reads are mostly cache hits, the shared-memory kernel may be *no faster* on a T4. Shared memory pays off when re-reads would otherwise go to DRAM: wider stencils, 3D, or GPUs with smaller caches. Whichever way the measurement comes out, explaining it with this model is the result.

## 3. Correctness strategy

| Check | How |
|---|---|
| OpenMP, MPI (1–4 ranks, blocking and overlapped) = serial | Bit-identical outputs. `-ffp-contract=off` stops the compiler from fusing multiply-adds differently in different builds. |
| GPU kernel logic = serial | `gas2d_emu` runs the kernels' per-thread code on the CPU with CUDA's block/thread/barrier semantics, on odd grid sizes (97×61, 33×130, 1×7) that leave partial blocks. Shared-memory tiles are filled with NaN first, so a halo cell that was never loaded shows up immediately. |
| Real GPU = serial | Within 10⁻⁴ of the peak (the GPU's fused multiply-add rounds differently). Run in the Colab notebook. |
| Physics | Analytic puff solution, first-order convergence, mass conservation, positivity (see `model.md`). |
| CUDA compiles | CI builds every `.cu` file with `nvcc` for sm_75 on each push. |
