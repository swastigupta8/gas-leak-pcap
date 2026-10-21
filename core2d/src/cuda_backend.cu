/*
 * cuda_backend.cu - the CUDA kernels and their host-side driver.
 *
 * Each kernel is a thin wrapper: it works out which cell(s) its thread owns and calls the shared
 * per-thread function from kernels.h, the same code the CPU emulator tests.
 *
 * Launch shape: 32 x 8 threads per block. 32 threads along x = one warp per tile row, so the
 * warp's 32 loads of consecutive floats combine into one 128-byte memory transaction
 * ("coalescing"). 256 threads per block lets several blocks share a streaming multiprocessor,
 * so while one block waits for memory another can compute ("latency hiding").
 */
#include "cuda_backend.h"
#include "kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        cudaError_t e_ = (call);                                                                 \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__,       \
                    __LINE__, cudaGetErrorString(e_));                                           \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

/* __restrict__ promises the compiler `in` and `out` never overlap, which lets it route reads of
 * `in` through the read-only data cache. */
__global__ void k_naive(const float *__restrict__ in, float *__restrict__ out, int nx, int ny, g2_weights k,
                        long long src, float s) {
    g2_naive_cell(in, out, nx, ny, k, src, s, blockIdx.x * G2_TILE_X + threadIdx.x,
                  blockIdx.y * G2_TILE_Y + threadIdx.y);
}

__global__ void k_shared(const float *__restrict__ in, float *__restrict__ out, int nx, int ny, g2_weights k,
                         long long src, float s) {
    __shared__ float tile[(G2_TILE_Y + 2) * G2_TILE_LD];
    /* No thread may return before the barrier: every thread must arrive at __syncthreads(). The
     * bounds checks live inside the two phases for that reason. */
    g2_tile_load(in, tile, nx, ny, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    __syncthreads();
    g2_tile_compute(tile, out, nx, ny, k, src, s, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

__global__ void k_coarse(const float *__restrict__ in, float *__restrict__ out, int nx, int ny, g2_weights k,
                         long long src, float s) {
    g2_coarse_column(in, out, nx, ny, k, src, s, blockIdx.x * G2_TILE_X + threadIdx.x,
                     (blockIdx.y * G2_TILE_Y + threadIdx.y) * G2_COARSE_Y);
}

__global__ void k_frame(const float *__restrict__ C, float *__restrict__ img, int nx, int ny, int fx, int fy) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x, oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox < fx && oy < fy) img[(size_t)oy * fx + ox] = g2_frame_pixel(C, nx, ny, fx, fy, ox, oy);
}

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

int g2_kernel_from_name(const char *name) {
    if (strcmp(name, "naive") == 0) return G2_NAIVE;
    if (strcmp(name, "shared") == 0) return G2_SHARED;
    if (strcmp(name, "coarse") == 0) return G2_COARSE;
    return -1;
}

int g2_gpu_create(g2_gpu *g, int nx, int ny) {
    memset(g, 0, sizeof *g);
    g->nx = nx;
    g->ny = ny;
    g->n = (size_t)(nx + 2) * (ny + 2);
    if (cudaMalloc(&g->a, g->n * sizeof(float)) != cudaSuccess) return -1;
    if (cudaMalloc(&g->b, g->n * sizeof(float)) != cudaSuccess) return -1;
    g2_gpu_zero(g);
    return 0;
}

void g2_gpu_zero(g2_gpu *g) {
    /* Both buffers, including the ghost ring the kernels never write. */
    CUDA_CHECK(cudaMemset(g->a, 0, g->n * sizeof(float)));
    CUDA_CHECK(cudaMemset(g->b, 0, g->n * sizeof(float)));
}

void g2_gpu_destroy(g2_gpu *g) {
    cudaFree(g->a);
    cudaFree(g->b);
    cudaFree(g->frame);
    memset(g, 0, sizeof *g);
}

void g2_gpu_upload(g2_gpu *g, const float *h) {
    CUDA_CHECK(cudaMemcpy(g->a, h, g->n * sizeof(float), cudaMemcpyHostToDevice));
}

void g2_gpu_download(const g2_gpu *g, float *h) {
    CUDA_CHECK(cudaMemcpy(h, g->a, g->n * sizeof(float), cudaMemcpyDeviceToHost));
}

float g2_gpu_run(g2_gpu *g, g2_kernel kernel, int steps, g2_weights k, long long src, float s) {
    dim3 block(G2_TILE_X, G2_TILE_Y);
    dim3 grid(div_up(g->nx, G2_TILE_X), div_up(g->ny, G2_TILE_Y));
    if (kernel == G2_COARSE) grid.y = div_up(g->ny, G2_TILE_Y * G2_COARSE_Y);

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int n = 0; n < steps; ++n) {
        /* One launch per step: the launch boundary is the global barrier between time steps
         * (every cell of step n must exist before any cell of step n+1 reads it). */
        switch (kernel) {
        case G2_NAIVE: k_naive<<<grid, block>>>(g->a, g->b, g->nx, g->ny, k, src, s); break;
        case G2_SHARED: k_shared<<<grid, block>>>(g->a, g->b, g->nx, g->ny, k, src, s); break;
        case G2_COARSE: k_coarse<<<grid, block>>>(g->a, g->b, g->nx, g->ny, k, src, s); break;
        }
        float *t = g->a; g->a = g->b; g->b = t;
    }
    CUDA_CHECK(cudaGetLastError()); /* catches bad launch configurations */
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms;
}

void g2_gpu_frame(g2_gpu *g, float *host, int fx, int fy) {
    if (fx * fy > g->frame_cap) {
        cudaFree(g->frame);
        CUDA_CHECK(cudaMalloc(&g->frame, sizeof(float) * fx * fy));
        g->frame_cap = fx * fy;
    }
    dim3 block(16, 16), grid(div_up(fx, 16), div_up(fy, 16));
    k_frame<<<grid, block>>>(g->a, g->frame, g->nx, g->ny, fx, fy);
    CUDA_CHECK(cudaGetLastError());
    /* Copying only the small image, not the whole field, keeps the PCIe transfer tiny. */
    CUDA_CHECK(cudaMemcpy(host, g->frame, sizeof(float) * fx * fy, cudaMemcpyDeviceToHost));
}

double g2_gpu_copy_gbps(g2_gpu *g, int reps) {
    const size_t bytes = g->n * sizeof(float);
    /* Copies the current field (a) into the spare buffer (b); `a` is not modified. */
    CUDA_CHECK(cudaMemcpy(g->b, g->a, bytes, cudaMemcpyDeviceToDevice)); /* warm-up */
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) CUDA_CHECK(cudaMemcpy(g->b, g->a, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    /* Each copy reads `bytes` and writes `bytes`. */
    return ms > 0 ? 2.0 * (double)bytes * reps / (ms * 1e6) : 0.0;
}

void g2_gpu_info(char *name, int len, double *peak_gbps) {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    snprintf(name, len, "%s", prop.name);
    int clk_khz = 0, bus_bits = 0;
    *peak_gbps = 0.0;
    /* Theoretical bandwidth = 2 (double data rate) x memory clock x bus width. */
    if (cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev) == cudaSuccess)
        *peak_gbps = 2.0 * clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
    cudaGetLastError(); /* clear any error from an unsupported attribute */
}
