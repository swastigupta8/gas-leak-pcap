/*
 * openmp.c - shared-memory parallel step.
 *
 * Each thread takes one contiguous block of rows (the same split schedule(static) would make)
 * and sweeps it with g2_step_rect, the serial kernel. Every cell costs the same, so equal blocks
 * give perfect load balance with zero scheduling overhead, and each thread streams through its
 * own region of memory.
 *
 * Why not simply `#pragma omp parallel for` around the loops? We measured it: with GCC the
 * outlined parallel loop lost the `restrict` (no-aliasing) guarantee, the inner loop stopped
 * vectorising, and the 1-thread OpenMP build ran ~1.5x SLOWER than the serial one. Calling the
 * already-vectorised serial sweep from each thread avoids that and keeps results bit-identical.
 *
 * There is no race: threads only read `in` and each writes only its own rows of `out`. The two
 * arrays are swapped by the caller between steps, so no lock or barrier is needed within a step
 * (the implicit barrier at the end of the parallel region separates consecutive steps).
 */
#include "gas2d.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void g2_step_omp(const float *in, float *out, int nx, int ny, g2_weights k) {
    const int ld = nx + 2;
#pragma omp parallel
    {
#ifdef _OPENMP
        const int t = omp_get_thread_num(), nt = omp_get_num_threads();
#else
        const int t = 0, nt = 1;
#endif
        /* Rows [j0, j1) for this thread; the blocks differ in size by at most one row. */
        const int j0 = 1 + (int)((long long)ny * t / nt);
        const int j1 = 1 + (int)((long long)ny * (t + 1) / nt);
        g2_step_rect(in, out, ld, 1, nx + 1, j0, j1, k);
    }
}
