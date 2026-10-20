/*
 * cuda_emu.h - runs the GPU kernels' per-thread code (kernels.h) on the CPU, following CUDA's
 * execution model:
 *
 *   grid of blocks  ->  outer loops over (bx, by)
 *   block of threads ->  inner loops over (tx, ty)
 *   __syncthreads() ->  finish phase 1 for EVERY thread of the block before any thread starts
 *                       phase 2, which is precisely the guarantee the barrier gives on a GPU
 *
 * If a kernel reads a halo cell that no thread loaded, or writes outside the grid, the emulated
 * result differs from the serial result and tests/test_correctness.py fails. That catches the
 * classic tiling bugs before the code ever reaches a GPU.
 */
#ifndef G2_CUDA_EMU_H
#define G2_CUDA_EMU_H

#include "kernels.h"

static inline int g2_div_up(int a, int b) { return (a + b - 1) / b; }

static void emu_naive(const float *in, float *out, int nx, int ny, g2_weights k, long long src, float s) {
    const int gx = g2_div_up(nx, G2_TILE_X), gy = g2_div_up(ny, G2_TILE_Y);
    for (int by = 0; by < gy; ++by)
        for (int bx = 0; bx < gx; ++bx)
            for (int ty = 0; ty < G2_TILE_Y; ++ty)
                for (int tx = 0; tx < G2_TILE_X; ++tx)
                    g2_naive_cell(in, out, nx, ny, k, src, s, bx * G2_TILE_X + tx, by * G2_TILE_Y + ty);
}

static void emu_shared(const float *in, float *out, int nx, int ny, g2_weights k, long long src, float s) {
    const int gx = g2_div_up(nx, G2_TILE_X), gy = g2_div_up(ny, G2_TILE_Y);
    float tile[(G2_TILE_Y + 2) * G2_TILE_LD];
    for (int by = 0; by < gy; ++by)
        for (int bx = 0; bx < gx; ++bx) {
            /* Poison the tile first: a halo cell that should have been loaded but was not would
             * otherwise silently reuse the previous block's value. NaN makes it visible. */
            for (int i = 0; i < (G2_TILE_Y + 2) * G2_TILE_LD; ++i) tile[i] = __builtin_nanf("");
            for (int ty = 0; ty < G2_TILE_Y; ++ty)
                for (int tx = 0; tx < G2_TILE_X; ++tx) g2_tile_load(in, tile, nx, ny, bx, by, tx, ty);
            /* ---- __syncthreads() ---- */
            for (int ty = 0; ty < G2_TILE_Y; ++ty)
                for (int tx = 0; tx < G2_TILE_X; ++tx) g2_tile_compute(tile, out, nx, ny, k, src, s, bx, by, tx, ty);
        }
}

static void emu_coarse(const float *in, float *out, int nx, int ny, g2_weights k, long long src, float s) {
    const int gx = g2_div_up(nx, G2_TILE_X), gy = g2_div_up(ny, G2_TILE_Y * G2_COARSE_Y);
    for (int by = 0; by < gy; ++by)
        for (int bx = 0; bx < gx; ++bx)
            for (int ty = 0; ty < G2_TILE_Y; ++ty)
                for (int tx = 0; tx < G2_TILE_X; ++tx)
                    g2_coarse_column(in, out, nx, ny, k, src, s, bx * G2_TILE_X + tx,
                                     (by * G2_TILE_Y + ty) * G2_COARSE_Y);
}

#endif /* G2_CUDA_EMU_H */
