// sim.cu — the simulator as a shared library, called from Python (server.py) through ctypes.
//
// Build for the GPU:  nvcc -O3 -arch=sm_75 -Xcompiler -fPIC -shared sim.cu -o libsim_cuda.so
// Build for the CPU:  g++ -O3 -march=native -fopenmp -fPIC -shared -x c++ sim.cu -o libsim_cpu.so
// (same source file: compat.h decides whether kernels run on GPU threads or CPU loops)
#include "kernels.cuh"
#include <utility>

struct Sim {
    int nx = 0, ny = 0, nz = 0; size_t N = 0;
    float *fa = nullptr, *fb = nullptr;             // air packets, ping-pong
    float *Ca = nullptr, *Cb = nullptr;             // gas concentration (ppm), ping-pong
    float *ux = nullptr, *uy = nullptr, *uz = nullptr, *nut = nullptr;
    uint8_t* solid = nullptr;                       // 1 = terrain or building
    int *ground = nullptr, *breath = nullptr;       // per column: first air box above terrain / above roofs
    float* gl = nullptr; float* wind = nullptr; uint8_t* vol = nullptr; int8_t* wvol = nullptr;
    FlowParams F{}; GasParams G{};
    Timer t1, t2;
};
static Sim S;

static dim3 lbm_block() { return dim3(64, 4, 1); }
static dim3 lbm_grid() { return dim3((S.nx + 63) / 64, (S.ny + 3) / 4, S.nz); }

extern "C" {

const char* sim_backend() { return BACKEND_NAME; }

int sim_create(int nx, int ny, int nz) {
    S.nx = nx; S.ny = ny; S.nz = nz; S.N = (size_t)nx * ny * nz;
    size_t N = S.N, cols = (size_t)nx * ny;
    CUDA_CHECK(cudaMalloc(&S.fa, 19 * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S.fb, 19 * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S.Ca, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S.Cb, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S.ux, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S.uy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S.uz, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S.nut, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S.solid, N));
    CUDA_CHECK(cudaMalloc(&S.ground, cols * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&S.breath, cols * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&S.gl, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S.wind, cols * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&S.vol, N));
    CUDA_CHECK(cudaMalloc(&S.wvol, 3 * N));
    cudaMemset(S.Ca, 0, N * sizeof(float)); cudaMemset(S.Cb, 0, N * sizeof(float));
    cudaMemset(S.ux, 0, N * sizeof(float)); cudaMemset(S.uy, 0, N * sizeof(float));
    cudaMemset(S.uz, 0, N * sizeof(float)); cudaMemset(S.nut, 0, N * sizeof(float));
    S.F.nx = S.G.nx = nx; S.F.ny = S.G.ny = ny; S.F.nz = S.G.nz = nz;
    S.F.gy0 = S.G.gy0 = 0; S.F.gny = S.G.gny = ny;
    size_t mb = (40 * N * 4 + 2 * N) >> 20;
    printf("[sim] backend=%s grid %d x %d x %d = %.1f M boxes, ~%zu MB\n", BACKEND_NAME, nx, ny, nz, N / 1e6, mb);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

void sim_set_geometry(const uint8_t* solid_h, const int* ground_h, const int* breath_h) {
    size_t cols = (size_t)S.nx * S.ny;
    cudaMemcpy(S.solid, solid_h, S.N, cudaMemcpyHostToDevice);
    cudaMemcpy(S.ground, ground_h, cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(S.breath, breath_h, cols * sizeof(int), cudaMemcpyHostToDevice);
}

void sim_set_flow(float tau0, float cs, float u10, float p_exp, float dirx, float diry, float ref_cells,
                  float ucap_profile, float sponge_w, float sponge_max, float g_beta, float ucap) {
    FlowParams& F = S.F;
    F.tau0 = tau0; F.cs2 = cs * cs; F.u10 = u10; F.p_exp = p_exp; F.dirx = dirx; F.diry = diry;
    F.ref_cells = ref_cells; F.ucap_profile = ucap_profile; F.sponge_w = sponge_w;
    F.sponge_max = sponge_max; F.g_beta = g_beta; F.ucap = ucap;
}

void sim_set_gas(float sc_inv, float kbg_h, float kbg_v, float kv_factor, float kmax, float umax, float decay) {
    GasParams& G = S.G;
    G.sc_inv = sc_inv; G.kbg_h = kbg_h; G.kbg_v = kbg_v; G.kv_factor = kv_factor;
    G.kmax = kmax; G.umax = umax; G.decay = decay;
}

// xyz = n*3 box coordinates, rate = ppm added per step at each
void sim_set_sources(int n, const int* xyz, const float* rate) {
    if (n > MAX_SRC) n = MAX_SRC;
    GasSource s[MAX_SRC];
    for (int k = 0; k < n; ++k) s[k] = GasSource{xyz[3 * k], xyz[3 * k + 1], xyz[3 * k + 2], rate[k]};
    if (n > 0) cudaMemcpyToSymbol(c_src, s, n * sizeof(GasSource));
    S.G.nsrc = n;
}

void sim_init_flow() {
    LAUNCH(init_flow, lbm_grid(), lbm_block(), S.fa, S.solid, S.ground, S.F);
    cudaDeviceSynchronize();
}

void sim_clear_gas() {
    cudaMemset(S.Ca, 0, S.N * sizeof(float)); cudaMemset(S.Cb, 0, S.N * sizeof(float));
}

// Advance n steps. gas_mode: 0 = naive gas kernel, 1 = shared-memory gas kernel, -1 = air only.
// ms_out[0] = air ms per step, ms_out[1] = gas ms per step.
void sim_step(int n, int gas_mode, float* ms_out) {
    float air = 0, gas = 0;
    dim3 gb(32, 8, 1), gg((S.nx + 31) / 32, (S.ny + 7) / 8, 1);
    for (int s = 0; s < n; ++s) {
        S.t1.start();
        LAUNCH(lbm_step, lbm_grid(), lbm_block(), S.fa, S.fb, S.solid, S.ground, S.Ca,
               S.ux, S.uy, S.uz, S.nut, S.F);
        air += S.t1.stop_ms();
        std::swap(S.fa, S.fb);
        if (gas_mode >= 0) {
            S.t2.start();
            if (gas_mode == 1)
                LAUNCH_SYNC(gas_step_shared, gg, gb, S.Ca, S.Cb, S.solid, S.ux, S.uy, S.uz, S.nut, S.G);
            else
                LAUNCH(gas_step_naive, lbm_grid(), lbm_block(), S.Ca, S.Cb, S.solid, S.ux, S.uy, S.uz, S.nut, S.G);
            gas += S.t2.stop_ms();
            std::swap(S.Ca, S.Cb);
        }
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) fprintf(stderr, "[sim] CUDA error: %s\n", cudaGetErrorString(e));
    if (ms_out) { ms_out[0] = air / (n > 0 ? n : 1); ms_out[1] = gas / (n > 0 ? n : 1); }
}

void sim_ground(float* out) {
    dim3 b(16, 16), g((S.nx + 15) / 16, (S.ny + 15) / 16);
    LAUNCH(ground_level, g, b, S.Ca, S.breath, S.gl, S.nx, S.ny);
    cudaMemcpy(out, S.gl, (size_t)S.nx * S.ny * sizeof(float), cudaMemcpyDeviceToHost);
}

void sim_volume(uint8_t* out, int fx, int fy, int fz) {
    int vx = (S.nx + fx - 1) / fx, vy = (S.ny + fy - 1) / fy, vz = (S.nz + fz - 1) / fz;
    dim3 b(16, 16), g((vx + 15) / 16, (vy + 15) / 16, vz);
    LAUNCH(volume_bytes, g, b, S.Ca, S.vol, S.nx, S.ny, S.nz, fx, fy, fz, vx, vy, vz);
    cudaMemcpy(out, S.vol, (size_t)vx * vy * vz, cudaMemcpyDeviceToHost);
}

void sim_wind(float* out, int f, int zoff) {
    int wx = (S.nx + f - 1) / f, wy = (S.ny + f - 1) / f;
    dim3 b(16, 16), g((wx + 15) / 16, (wy + 15) / 16);
    LAUNCH(wind_slice, g, b, S.ux, S.uy, S.ground, S.wind, S.nx, S.ny, S.nz, f, zoff, wx, wy);
    cudaMemcpy(out, S.wind, (size_t)wx * wy * 2 * sizeof(float), cudaMemcpyDeviceToHost);
}

// 3D wind shrunk by (fx, fy, fz) as int8 in units of 1/scale lattice speed (server passes scale = 10*dx/dt -> 0.1 m/s)
void sim_wind_volume(int8_t* out, int fx, int fy, int fz, float scale) {
    int vx = (S.nx + fx - 1) / fx, vy = (S.ny + fy - 1) / fy, vz = (S.nz + fz - 1) / fz;
    dim3 b(16, 16), g((vx + 15) / 16, (vy + 15) / 16, vz);
    LAUNCH(wind_volume, g, b, S.ux, S.uy, S.uz, S.solid, S.wvol, S.nx, S.ny, S.nz, fx, fy, fz, vx, vy, vz, scale);
    cudaMemcpy(out, S.wvol, (size_t)3 * vx * vy * vz, cudaMemcpyDeviceToHost);
}

// Full fields, for validation and plots
void sim_get_C(float* out) { cudaMemcpy(out, S.Ca, S.N * sizeof(float), cudaMemcpyDeviceToHost); }
void sim_get_u(float* ux, float* uy, float* uz) {
    cudaMemcpy(ux, S.ux, S.N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(uy, S.uy, S.N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(uz, S.uz, S.N * 4, cudaMemcpyDeviceToHost);
}

void sim_set_threads(int n) {
#ifndef __CUDACC__
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}

void sim_free() {
    float* fs[] = {S.fa, S.fb, S.Ca, S.Cb, S.ux, S.uy, S.uz, S.nut, S.gl, S.wind};
    for (float* p : fs) cudaFree(p);
    cudaFree(S.solid); cudaFree(S.ground); cudaFree(S.breath); cudaFree(S.vol); cudaFree(S.wvol);
    S = Sim();
}

}  // extern "C"
