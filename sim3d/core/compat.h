// compat.h — lets the SAME kernel code run on an NVIDIA GPU (compiled with nvcc)
// or on the CPU (compiled with g++), where each "GPU thread" becomes one loop iteration.
//
//   nvcc  -> real CUDA: kernels run on thousands of GPU threads at once
//   g++   -> CPU emulation: blocks are spread over CPU cores with OpenMP, threads run in a loop
//
// This gives a fair CPU-vs-GPU speedup (identical maths) and lets the physics be tested
// on a laptop without a GPU.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#ifdef __CUDACC__
// ============================== real CUDA ==================================
#include <cuda_runtime.h>
#define HD __host__ __device__ __forceinline__
#define DEV __device__ __forceinline__
#define LAUNCH(kernel, grid, block, ...)      kernel<<<grid, block>>>(__VA_ARGS__)
#define LAUNCH_SYNC(kernel, grid, block, ...) kernel<<<grid, block>>>(__VA_ARGS__)
#define SYNC_THREADS() __syncthreads()
#define BACKEND_NAME "cuda"

#else
// ============================== CPU emulation ==============================
#include <chrono>
#include <omp.h>
#define HD inline
#define DEV inline
#define __global__
#define __device__
#define __host__
#define __constant__ static
#define __shared__ static
#define __forceinline__ inline
#define __restrict__ __restrict
#define BACKEND_NAME "cpu"

struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
struct uint3e { unsigned x, y, z; };
inline thread_local uint3e threadIdx, blockIdx;
inline thread_local dim3 blockDim, gridDim;

// Kernels WITHOUT __syncthreads: every block can run on any core, threads in a plain loop.
template <class K, class... A>
inline void emu_launch(K k, dim3 g, dim3 b, A... a) {
    long nb = (long)g.x * g.y * g.z;
    #pragma omp parallel for schedule(dynamic, 4)
    for (long n = 0; n < nb; ++n) {
        blockIdx = {unsigned(n % g.x), unsigned((n / g.x) % g.y), unsigned(n / ((long)g.x * g.y))};
        gridDim = g; blockDim = b;
        for (unsigned tz = 0; tz < b.z; ++tz)
            for (unsigned ty = 0; ty < b.y; ++ty)
                for (unsigned tx = 0; tx < b.x; ++tx) { threadIdx = {tx, ty, tz}; k(a...); }
    }
}
// Kernels WITH __syncthreads (shared memory): one block at a time, one OS thread per CUDA
// thread, __syncthreads() becomes an OpenMP barrier. Slow; only used for correctness tests.
template <class K, class... A>
inline void emu_launch_sync(K k, dim3 g, dim3 b, A... a) {
    int nt = b.x * b.y * b.z;
    omp_set_dynamic(0);
    for (long n = 0; n < (long)g.x * g.y * g.z; ++n) {
        uint3e bi = {unsigned(n % g.x), unsigned((n / g.x) % g.y), unsigned(n / ((long)g.x * g.y))};
        #pragma omp parallel num_threads(nt)
        {
            int t = omp_get_thread_num();
            threadIdx = {unsigned(t % b.x), unsigned((t / b.x) % b.y), unsigned(t / (b.x * b.y))};
            blockIdx = bi; blockDim = b; gridDim = g;
            k(a...);
        }
    }
}
inline void emu_barrier() {
    #pragma omp barrier
}
#define LAUNCH(kernel, grid, block, ...)      emu_launch(kernel, grid, block, __VA_ARGS__)
#define LAUNCH_SYNC(kernel, grid, block, ...) emu_launch_sync(kernel, grid, block, __VA_ARGS__)
#define SYNC_THREADS() emu_barrier()

// Memory API: on the CPU "device memory" is just normal memory.
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
inline cudaError_t cudaMalloc(void* p, size_t n) { *(void**)p = std::calloc(1, n); return *(void**)p ? 0 : 1; }
inline cudaError_t cudaFree(void* p) { std::free(p); return 0; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, int) { std::memcpy(d, s, n); return 0; }
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return 0; }
#define cudaMemcpyToSymbol(sym, src, n) std::memcpy(&(sym), (src), (n))
inline cudaError_t cudaDeviceSynchronize() { return 0; }
inline cudaError_t cudaGetLastError() { return 0; }
inline const char* cudaGetErrorString(cudaError_t) { return "no error"; }
#endif

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); } while (0)

// Wall-clock timer that works for both backends (waits for the GPU before reading the clock).
struct Timer {
#ifdef __CUDACC__
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop_ms() { cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
#else
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    float stop_ms() { return std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count(); }
#endif
};
