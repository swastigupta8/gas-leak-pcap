/*
 * cuda_backend.h - host-side control of the GPU simulation (memory, launches, timing).
 * Used by the benchmark executable (cuda_main.cu) and the dashboard library (lib_cuda.cu).
 */
#ifndef G2_CUDA_BACKEND_H
#define G2_CUDA_BACKEND_H

#include "gas2d.h"

enum g2_kernel { G2_NAIVE = 0, G2_SHARED = 1, G2_COARSE = 2 };

struct g2_gpu {
    int nx, ny;
    size_t n;     /* padded cell count */
    float *a, *b; /* device ping-pong buffers; `a` always holds the current field */
    float *frame; /* device buffer for dashboard frames */
    int frame_cap;
};

/* Returns 0 on success. Both buffers start at zero (clean air, zero ghost ring). */
int g2_gpu_create(g2_gpu *g, int nx, int ny);
void g2_gpu_destroy(g2_gpu *g);
void g2_gpu_upload(g2_gpu *g, const float *host_padded);
void g2_gpu_download(const g2_gpu *g, float *host_padded);
void g2_gpu_zero(g2_gpu *g);

/* Run `steps` steps; returns elapsed GPU time in ms (measured with CUDA events). */
float g2_gpu_run(g2_gpu *g, g2_kernel kernel, int steps, g2_weights k, long long src, float s);

/* Max-pooled fx x fy image of the current field into host memory (row-major, south row first). */
void g2_gpu_frame(g2_gpu *g, float *host, int fx, int fy);

/* Name of the GPU and its theoretical memory bandwidth in GB/s (0 if unknown). */
void g2_gpu_info(char *name, int len, double *peak_gbps);

int g2_kernel_from_name(const char *name); /* -1 if unknown */

#endif
