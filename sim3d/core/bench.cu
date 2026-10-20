// bench.cu — speed and correctness measurements for the report.
//
// GPU build:  nvcc -O3 -arch=sm_75 bench.cu -o bench_cuda
// CPU build:  g++ -O3 -march=native -fopenmp -x c++ bench.cu -o bench_cpu
//
// Usage: ./bench_cuda <nx> <ny> <nz> <steps> [threads]     -> one JSON line on stdout
//        ./bench_cpu  check                                 -> writes check_cpu.bin  (same for cuda)
#include "testcase.h"
#include <string>
#include <utility>

struct Fields {
    float *fa, *fb, *Ca, *Cb, *ux, *uy, *uz, *nut; uint8_t* solid; int* ground;
};

static Fields alloc(const TestCase& T) {
    Fields D; size_t N = (size_t)T.nx * T.ny * T.nz;
    cudaMalloc(&D.fa, 19 * N * 4); cudaMalloc(&D.fb, 19 * N * 4);
    cudaMalloc(&D.Ca, N * 4); cudaMalloc(&D.Cb, N * 4);
    cudaMalloc(&D.ux, N * 4); cudaMalloc(&D.uy, N * 4); cudaMalloc(&D.uz, N * 4); cudaMalloc(&D.nut, N * 4);
    cudaMalloc(&D.solid, N); cudaMalloc(&D.ground, (size_t)T.nx * T.ny * 4);
    cudaMemcpy(D.solid, T.solid.data(), N, cudaMemcpyHostToDevice);
    cudaMemcpy(D.ground, T.ground.data(), (size_t)T.nx * T.ny * 4, cudaMemcpyHostToDevice);
    cudaMemset(D.Ca, 0, N * 4); cudaMemset(D.Cb, 0, N * 4);
    cudaMemset(D.ux, 0, N * 4); cudaMemset(D.uy, 0, N * 4); cudaMemset(D.uz, 0, N * 4); cudaMemset(D.nut, 0, N * 4);
    cudaMemcpyToSymbol(c_src, &T.src, sizeof(GasSource));
    dim3 b(64, 4, 1), g((T.nx + 63) / 64, (T.ny + 3) / 4, T.nz);
    LAUNCH(init_flow, g, b, D.fa, D.solid, D.ground, T.F);
    cudaDeviceSynchronize();
    return D;
}
static void release(Fields& D) {
    float* fs[] = {D.fa, D.fb, D.Ca, D.Cb, D.ux, D.uy, D.uz, D.nut};
    for (float* p : fs) cudaFree(p);
    cudaFree(D.solid); cudaFree(D.ground);
}

// one coupled step: air then gas. mode 0 = naive gas, 1 = shared-memory gas
static void step(const TestCase& T, Fields& D, int mode, float* t_air, float* t_gas) {
    dim3 b(64, 4, 1), g((T.nx + 63) / 64, (T.ny + 3) / 4, T.nz);
    dim3 sb(32, 8, 1), sg((T.nx + 31) / 32, (T.ny + 7) / 8, 1);
    Timer ta, tg;
    ta.start();
    LAUNCH(lbm_step, g, b, D.fa, D.fb, D.solid, D.ground, D.Ca, D.ux, D.uy, D.uz, D.nut, T.F);
    float a = ta.stop_ms();
    std::swap(D.fa, D.fb);
    tg.start();
    if (mode == 1) LAUNCH_SYNC(gas_step_shared, sg, sb, D.Ca, D.Cb, D.solid, D.ux, D.uy, D.uz, D.nut, T.G);
    else           LAUNCH(gas_step_naive, g, b, D.Ca, D.Cb, D.solid, D.ux, D.uy, D.uz, D.nut, T.G);
    float c = tg.stop_ms();
    std::swap(D.Ca, D.Cb);
    if (t_air) *t_air += a;
    if (t_gas) *t_gas += c;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "check") {
        // small fixed run whose result is compared between the CPU and GPU builds
        TestCase T = make_testcase(64, 48, 12);
        Fields D = alloc(T);
        for (int s = 0; s < 120; ++s) step(T, D, 0, nullptr, nullptr);
        size_t N = (size_t)T.nx * T.ny * T.nz;
        std::vector<float> C(N), U(N);
        cudaMemcpy(C.data(), D.Ca, N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(U.data(), D.ux, N * 4, cudaMemcpyDeviceToHost);
        std::string fn = std::string("check_") + BACKEND_NAME + ".bin";
        FILE* f = fopen(fn.c_str(), "wb"); fwrite(C.data(), 4, N, f); fwrite(U.data(), 4, N, f); fclose(f);
        double tot = 0; for (float v : C) tot += v;
        printf("{\"backend\": \"%s\", \"check_file\": \"%s\", \"total_ppm\": %.3f}\n", BACKEND_NAME, fn.c_str(), tot);
        release(D);
        return 0;
    }
    int nx = argc > 1 ? atoi(argv[1]) : 256, ny = argc > 2 ? atoi(argv[2]) : 256, nz = argc > 3 ? atoi(argv[3]) : 24;
    int steps = argc > 4 ? atoi(argv[4]) : 50;
#ifndef __CUDACC__
    int threads = argc > 5 ? atoi(argv[5]) : omp_get_max_threads();
    omp_set_num_threads(threads);
#else
    int threads = 0;
#endif
    TestCase T = make_testcase(nx, ny, nz);
    Fields D = alloc(T);
    for (int s = 0; s < 3; ++s) step(T, D, 0, nullptr, nullptr);          // warm-up
    float air = 0, gas_naive = 0, gas_shared = 0, dummy = 0;
    for (int s = 0; s < steps; ++s) step(T, D, 0, &air, &gas_naive);
    bool do_shared = true;
#ifndef __CUDACC__
    do_shared = false;     // the CPU emulation of shared memory is only for correctness, far too slow to time
#endif
    if (do_shared) for (int s = 0; s < steps; ++s) step(T, D, 1, &dummy, &gas_shared);
    double cells = (double)nx * ny * nz;
    printf("{\"backend\": \"%s\", \"threads\": %d, \"nx\": %d, \"ny\": %d, \"nz\": %d, \"steps\": %d, "
           "\"air_ms\": %.4f, \"gas_naive_ms\": %.4f, \"gas_shared_ms\": %.4f, \"air_mlups\": %.1f}\n",
           BACKEND_NAME, threads, nx, ny, nz, steps, air / steps, gas_naive / steps,
           do_shared ? gas_shared / steps : 0.0, cells / (air / steps * 1e-3) / 1e6);
    release(D);
    return 0;
}
