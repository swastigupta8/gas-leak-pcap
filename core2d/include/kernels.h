/*
 * kernels.h - the per-thread work of every GPU kernel, written as plain functions.
 *
 * The real CUDA kernels (src/cuda_backend.cu) are thin wrappers that compute their thread's
 * coordinates and call these functions. The CPU emulator (src/emu_main.c + include/cuda_emu.h)
 * calls the SAME functions in loops that reproduce CUDA's execution model, so the indexing,
 * bounds checks and shared-memory tiling are tested on any laptop, without a GPU.
 *
 * Three kernels, all one launch per time step:
 *
 *  1. naive   - one thread per cell, all five values read from global memory. Neighbouring
 *               threads re-read each other's cells; on modern GPUs the L1/L2 caches absorb
 *               much of that re-reading, which is exactly what the benchmark measures.
 *
 *  2. shared  - each block first copies its tile of cells PLUS a one-cell halo into on-chip
 *               shared memory, waits (__syncthreads), then computes from the tile. Every value
 *               is read from global memory once per block instead of up to five times.
 *
 *  3. coarse  - "thread coarsening": each thread updates G2_COARSE_Y cells in a column. It slides
 *               up the column keeping south, centre and north values in registers, so each step
 *               north reads ONE new value instead of three. Fewer threads, more work each, less
 *               memory traffic.
 *
 * Coordinates: (gi, gj) are 0-based interior coordinates; the padded array index of that cell is
 * g2_idx(gi + 1, gj + 1, nx + 2). `src` is the padded index of the leak cell, or -1 for none.
 */
#ifndef G2_KERNELS_H
#define G2_KERNELS_H

#include "gas2d.h"

#ifndef G2_TILE_X
#define G2_TILE_X 32 /* one warp wide: a row of the tile is one coalesced 128-byte read */
#endif
#ifndef G2_TILE_Y
#define G2_TILE_Y 8  /* 32 x 8 = 256 threads per block */
#endif
#ifndef G2_COARSE_Y
#define G2_COARSE_Y 8 /* cells per thread in the coarsened kernel */
#endif

#define G2_TILE_LD (G2_TILE_X + 2) /* shared tile row length including the halo */

/* ---- 1. naive ---------------------------------------------------------------------------- */
static inline G2_HD void g2_naive_cell(const float *in, float *out, int nx, int ny, g2_weights k,
                                       long long src, float s, int gi, int gj) {
    if (gi >= nx || gj >= ny) return; /* the last blocks overhang the grid */
    const int ld = nx + 2;
    const size_t c = g2_idx(gi + 1, gj + 1, ld);
    float val = g2_stencil(in[c], in[c + 1], in[c - 1], in[c + ld], in[c - ld], k);
    if ((long long)c == src) val += s;
    out[c] = val;
}

/* ---- 2. shared-memory tile ---------------------------------------------------------------- */
/* Phase 1, run by every thread (tx, ty) of block (bx, by): copy this thread's cell into the tile,
 * and if the thread sits on the tile edge, also the halo cell just outside it. Padded coordinates
 * run 0..nx+1, so anything beyond that does not exist and is skipped. */
static inline G2_HD void g2_tile_load(const float *in, float *tile, int nx, int ny, int bx, int by, int tx,
                                      int ty) {
    const int ld = nx + 2;
    const int px = bx * G2_TILE_X + tx + 1, py = by * G2_TILE_Y + ty + 1; /* padded coords */
    const int t = (ty + 1) * G2_TILE_LD + (tx + 1);
    const int okx = px <= nx + 1, oky = py <= ny + 1;
    tile[t] = (okx && oky) ? in[g2_idx(px, py, ld)] : 0.0f;
    if (tx == 0 && oky) tile[t - 1] = in[g2_idx(px - 1, py, ld)];
    if (tx == G2_TILE_X - 1 && oky && px + 1 <= nx + 1) tile[t + 1] = in[g2_idx(px + 1, py, ld)];
    if (ty == 0 && okx) tile[t - G2_TILE_LD] = in[g2_idx(px, py - 1, ld)];
    if (ty == G2_TILE_Y - 1 && okx && py + 1 <= ny + 1) tile[t + G2_TILE_LD] = in[g2_idx(px, py + 1, ld)];
}

/* Phase 2, after the barrier: compute from the tile only. */
static inline G2_HD void g2_tile_compute(const float *tile, float *out, int nx, int ny, g2_weights k,
                                         long long src, float s, int bx, int by, int tx, int ty) {
    const int gi = bx * G2_TILE_X + tx, gj = by * G2_TILE_Y + ty;
    if (gi >= nx || gj >= ny) return;
    const int t = (ty + 1) * G2_TILE_LD + (tx + 1);
    float val = g2_stencil(tile[t], tile[t + 1], tile[t - 1], tile[t + G2_TILE_LD], tile[t - G2_TILE_LD], k);
    const size_t c = g2_idx(gi + 1, gj + 1, nx + 2);
    if ((long long)c == src) val += s;
    out[c] = val;
}

/* ---- 3. coarsened column ------------------------------------------------------------------- */
/* Thread updates cells (gi, gj0) .. (gi, gj0 + G2_COARSE_Y - 1). */
static inline G2_HD void g2_coarse_column(const float *in, float *out, int nx, int ny, g2_weights k,
                                          long long src, float s, int gi, int gj0) {
    if (gi >= nx || gj0 >= ny) return;
    const int ld = nx + 2;
    size_t c = g2_idx(gi + 1, gj0 + 1, ld);
    float south = in[c - ld], centre = in[c];
    const int end = gj0 + G2_COARSE_Y < ny ? gj0 + G2_COARSE_Y : ny;
    for (int gj = gj0; gj < end; ++gj, c += ld) {
        const float north = in[c + ld]; /* the only new vertical read per cell */
        float val = g2_stencil(centre, in[c + 1], in[c - 1], north, south, k);
        if ((long long)c == src) val += s;
        out[c] = val;
        south = centre; /* slide the 3-value window one row north */
        centre = north;
    }
}

/* ---- dashboard frame ----------------------------------------------------------------------- */
/* The dashboard shows a small image (e.g. 500 x 500) of a big grid. Each image pixel takes the
 * MAXIMUM of the cells it covers rather than the average: averaging would hide a narrow toxic
 * plume, and for a safety display the worst value is the one that matters. */
static inline G2_HD float g2_frame_pixel(const float *C, int nx, int ny, int fx, int fy, int ox, int oy) {
    const int i0 = (int)((long long)ox * nx / fx), i1 = (int)((long long)(ox + 1) * nx / fx);
    const int j0 = (int)((long long)oy * ny / fy), j1 = (int)((long long)(oy + 1) * ny / fy);
    float m = 0.0f;
    for (int j = j0; j < (j1 > j0 ? j1 : j0 + 1); ++j)
        for (int i = i0; i < (i1 > i0 ? i1 : i0 + 1); ++i) {
            float v = C[g2_idx(i + 1, j + 1, nx + 2)];
            m = v > m ? v : m;
        }
    return m;
}

#endif /* G2_KERNELS_H */
