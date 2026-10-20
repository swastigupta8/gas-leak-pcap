// mpi_sim.cu — the same simulation split across several processes with MPI.
//
// The map is cut into horizontal strips (groups of rows), one per MPI rank. Each rank stores its
// strip plus one extra "halo" row on each side:
//
//      rank 1 :  [halo] [rows 40..79] [halo]
//      rank 0 :  [halo] [rows  0..39] [halo]
//
// After every air step, each rank sends its first and last real rows (all 19 air packets plus the
// velocity and turbulence they produce) to its neighbours, who store them in their halo rows.
// After every gas step it does the same for the gas concentration. Then every box at a strip edge
// sees exactly the neighbours it would see in a single big run, so the answer is identical.
//
// GPU build:  nvcc -O3 -arch=sm_75 -ccbin mpicxx mpi_sim.cu -o mpi_cuda
// CPU build:  mpicxx -O3 -march=native -fopenmp -x c++ mpi_sim.cu -o mpi_cpu
// Run:        mpirun -np 4 ./mpi_cuda 256 256 24 100
#include <mpi.h>
#include "testcase.h"
#include <utility>
#include <algorithm>

// copy one row (all z, all x) of `nf` fields into a flat buffer, and back
__global__ void pack_row(const float* src, float* buf, size_t fstride, int nx, int nyl, int nz, int row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, z = blockIdx.y, q = blockIdx.z;
    if (x >= nx) return;
    buf[((size_t)q * nz + z) * nx + x] = src[q * fstride + IDX(x, row, z, nx, nyl)];
}
__global__ void unpack_row(float* dst, const float* buf, size_t fstride, int nx, int nyl, int nz, int row) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, z = blockIdx.y, q = blockIdx.z;
    if (x >= nx) return;
    dst[q * fstride + IDX(x, row, z, nx, nyl)] = buf[((size_t)q * nz + z) * nx + x];
}

struct Rank {
    int rank, P, nx, ny, nz, y0, nyl, nyL;   // nyl = my real rows, nyL = nyl + 2 halo rows
    int up, down;                            // neighbour ranks (MPI_PROC_NULL at the map edge)
    size_t N;
    float *fa, *fb, *Ca, *Cb, *ux, *uy, *uz, *nut; uint8_t* solid; int* ground;
    float *dbuf; std::vector<float> s1, s2, r1, r2;
    FlowParams F; GasParams G;
    double t_comm = 0;
};

// swap halo rows of `nf` fields stored back to back with stride `fstride`
static void exchange(Rank& R, float* field, int nf, size_t fstride) {
    double t0 = MPI_Wtime();
    size_t row = (size_t)nf * R.nz * R.nx;
    dim3 b(128), g((R.nx + 127) / 128, R.nz, nf);
    // my first real row (local 1) goes DOWN to rank-1's top halo; my last real row goes UP
    LAUNCH(pack_row, g, b, field, R.dbuf, fstride, R.nx, R.nyL, R.nz, 1);
    cudaMemcpy(R.s1.data(), R.dbuf, row * 4, cudaMemcpyDeviceToHost);
    LAUNCH(pack_row, g, b, field, R.dbuf, fstride, R.nx, R.nyL, R.nz, R.nyl);
    cudaMemcpy(R.s2.data(), R.dbuf, row * 4, cudaMemcpyDeviceToHost);
    MPI_Sendrecv(R.s1.data(), (int)row, MPI_FLOAT, R.down, 0, R.r2.data(), (int)row, MPI_FLOAT, R.up, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(R.s2.data(), (int)row, MPI_FLOAT, R.up, 1, R.r1.data(), (int)row, MPI_FLOAT, R.down, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (R.down != MPI_PROC_NULL) {
        cudaMemcpy(R.dbuf, R.r1.data(), row * 4, cudaMemcpyHostToDevice);
        LAUNCH(unpack_row, g, b, field, R.dbuf, fstride, R.nx, R.nyL, R.nz, 0);
    }
    if (R.up != MPI_PROC_NULL) {
        cudaMemcpy(R.dbuf, R.r2.data(), row * 4, cudaMemcpyHostToDevice);
        LAUNCH(unpack_row, g, b, field, R.dbuf, fstride, R.nx, R.nyL, R.nz, R.nyl + 1);
    }
    cudaDeviceSynchronize();
    R.t_comm += MPI_Wtime() - t0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    Rank R; MPI_Comm_rank(MPI_COMM_WORLD, &R.rank); MPI_Comm_size(MPI_COMM_WORLD, &R.P);
    int nx = argc > 1 ? atoi(argv[1]) : 192, ny = argc > 2 ? atoi(argv[2]) : 192, nz = argc > 3 ? atoi(argv[3]) : 20;
    int steps = argc > 4 ? atoi(argv[4]) : 100;
    int gmode = argc > 5 ? atoi(argv[5]) : 0;
#ifdef __CUDACC__
    int ndev = 1; cudaGetDeviceCount(&ndev); cudaSetDevice(R.rank % (ndev > 0 ? ndev : 1));
#else
    omp_set_num_threads(1);
#endif
    TestCase T = make_testcase(nx, ny, nz);
    R.nx = nx; R.ny = ny; R.nz = nz;
    // split rows as evenly as possible
    R.nyl = ny / R.P + (R.rank < ny % R.P); R.y0 = 0;
    for (int r = 0; r < R.rank; ++r) R.y0 += ny / R.P + (r < ny % R.P);
    R.nyL = R.nyl + 2;
    R.down = R.rank > 0 ? R.rank - 1 : MPI_PROC_NULL;          // strip below (south)
    R.up = R.rank < R.P - 1 ? R.rank + 1 : MPI_PROC_NULL;      // strip above (north)
    R.N = (size_t)nx * R.nyL * nz;

    // local geometry: rows y0-1 .. y0+nyl (clamped at the map edge; those halo rows are never used)
    std::vector<uint8_t> sol(R.N); std::vector<int> gr((size_t)nx * R.nyL);
    for (int yl = 0; yl < R.nyL; ++yl) {
        int gy = std::min(std::max(R.y0 - 1 + yl, 0), ny - 1);
        for (int x = 0; x < nx; ++x) {
            gr[yl * nx + x] = T.ground[gy * nx + x];
            for (int z = 0; z < nz; ++z) sol[IDX(x, yl, z, nx, R.nyL)] = T.solid[IDX(x, gy, z, nx, ny)];
        }
    }
    R.F = T.F; R.F.ny = R.nyL; R.F.gy0 = R.y0 - 1; R.F.gny = ny;
    R.G = T.G; R.G.ny = R.nyL; R.G.gy0 = R.y0 - 1; R.G.gny = ny;
    cudaMemcpyToSymbol(c_src, &T.src, sizeof(GasSource));
    size_t N = R.N;
    cudaMalloc(&R.fa, 19 * N * 4); cudaMalloc(&R.fb, 19 * N * 4); cudaMalloc(&R.Ca, N * 4); cudaMalloc(&R.Cb, N * 4);
    cudaMalloc(&R.ux, N * 4); cudaMalloc(&R.uy, N * 4); cudaMalloc(&R.uz, N * 4); cudaMalloc(&R.nut, N * 4);
    cudaMalloc(&R.solid, N); cudaMalloc(&R.ground, (size_t)nx * R.nyL * 4);
    cudaMemcpy(R.solid, sol.data(), N, cudaMemcpyHostToDevice);
    cudaMemcpy(R.ground, gr.data(), (size_t)nx * R.nyL * 4, cudaMemcpyHostToDevice);
    cudaMemset(R.Ca, 0, N * 4); cudaMemset(R.Cb, 0, N * 4);
    cudaMemset(R.ux, 0, N * 4); cudaMemset(R.uy, 0, N * 4); cudaMemset(R.uz, 0, N * 4); cudaMemset(R.nut, 0, N * 4);
    size_t rowmax = (size_t)19 * nz * nx;
    cudaMalloc(&R.dbuf, rowmax * 4); R.s1.resize(rowmax); R.s2.resize(rowmax); R.r1.resize(rowmax); R.r2.resize(rowmax);

    dim3 b(64, 4, 1), g((nx + 63) / 64, (R.nyL + 3) / 4, nz);
    dim3 sb(32, 8, 1), sg((nx + 31) / 32, (R.nyL + 7) / 8, 1);
    LAUNCH(init_flow, g, b, R.fa, R.solid, R.ground, R.F);
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime(), t_comp = 0;
    for (int s = 0; s < steps; ++s) {
        double c0 = MPI_Wtime();
        LAUNCH(lbm_step, g, b, R.fa, R.fb, R.solid, R.ground, R.Ca, R.ux, R.uy, R.uz, R.nut, R.F);
        cudaDeviceSynchronize(); t_comp += MPI_Wtime() - c0;
        std::swap(R.fa, R.fb);
        exchange(R, R.fa, 19, N);                               // air packets
        exchange(R, R.ux, 1, N); exchange(R, R.uy, 1, N);       // velocities and turbulence the gas needs
        exchange(R, R.uz, 1, N); exchange(R, R.nut, 1, N);
        c0 = MPI_Wtime();
        if (gmode == 1) LAUNCH_SYNC(gas_step_shared, sg, sb, R.Ca, R.Cb, R.solid, R.ux, R.uy, R.uz, R.nut, R.G);
        else            LAUNCH(gas_step_naive, g, b, R.Ca, R.Cb, R.solid, R.ux, R.uy, R.uz, R.nut, R.G);
        cudaDeviceSynchronize(); t_comp += MPI_Wtime() - c0;
        std::swap(R.Ca, R.Cb);
        exchange(R, R.Ca, 1, N);                                // gas
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double total = MPI_Wtime() - t0;

    // ---- gather my real rows of C and ux to rank 0 ----
    size_t mine = (size_t)nz * R.nyl * nx;
    std::vector<float> C(N), U(N), packC(mine), packU(mine);
    cudaMemcpy(C.data(), R.Ca, N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(U.data(), R.ux, N * 4, cudaMemcpyDeviceToHost);
    for (int z = 0; z < nz; ++z) for (int yl = 0; yl < R.nyl; ++yl) for (int x = 0; x < nx; ++x) {
        size_t k = ((size_t)z * R.nyl + yl) * nx + x;
        packC[k] = C[IDX(x, yl + 1, z, nx, R.nyL)]; packU[k] = U[IDX(x, yl + 1, z, nx, R.nyL)];
    }
    std::vector<int> cnt(R.P), off(R.P), rows(R.P), y0s(R.P);
    int myrows = R.nyl;
    MPI_Gather(&myrows, 1, MPI_INT, rows.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int acc = 0;
    for (int r = 0; r < R.P; ++r) { cnt[r] = rows[r] * nx * nz; off[r] = acc; y0s[r] = (r ? y0s[r - 1] + rows[r - 1] : 0); acc += cnt[r]; }
    std::vector<float> allC, allU;
    if (R.rank == 0) { allC.resize((size_t)nx * ny * nz); allU.resize((size_t)nx * ny * nz); }
    MPI_Gatherv(packC.data(), (int)mine, MPI_FLOAT, allC.data(), cnt.data(), off.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(packU.data(), (int)mine, MPI_FLOAT, allU.data(), cnt.data(), off.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    double comm_max = 0, comp_max = 0;
    MPI_Reduce(&R.t_comm, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (R.rank == 0) {
        // the same run as ONE domain, for comparison
        size_t NG = (size_t)nx * ny * nz;
        float *fa, *fb, *Ca, *Cb, *ux, *uy, *uz, *nut; uint8_t* solid; int* ground;
        cudaMalloc(&fa, 19 * NG * 4); cudaMalloc(&fb, 19 * NG * 4); cudaMalloc(&Ca, NG * 4); cudaMalloc(&Cb, NG * 4);
        cudaMalloc(&ux, NG * 4); cudaMalloc(&uy, NG * 4); cudaMalloc(&uz, NG * 4); cudaMalloc(&nut, NG * 4);
        cudaMalloc(&solid, NG); cudaMalloc(&ground, (size_t)nx * ny * 4);
        cudaMemcpy(solid, T.solid.data(), NG, cudaMemcpyHostToDevice);
        cudaMemcpy(ground, T.ground.data(), (size_t)nx * ny * 4, cudaMemcpyHostToDevice);
        cudaMemset(Ca, 0, NG * 4); cudaMemset(Cb, 0, NG * 4);
        dim3 g1((nx + 63) / 64, (ny + 3) / 4, nz);
        LAUNCH(init_flow, g1, b, fa, solid, ground, T.F);
        for (int s = 0; s < steps; ++s) {
            LAUNCH(lbm_step, g1, b, fa, fb, solid, ground, Ca, ux, uy, uz, nut, T.F); std::swap(fa, fb);
            LAUNCH(gas_step_naive, g1, b, Ca, Cb, solid, ux, uy, uz, nut, T.G); std::swap(Ca, Cb);
        }
        std::vector<float> refC(NG), refU(NG);
        cudaMemcpy(refC.data(), Ca, NG * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(refU.data(), ux, NG * 4, cudaMemcpyDeviceToHost);
        double eC = 0, eU = 0, mC = 0;
        for (int r = 0; r < R.P; ++r)
            for (int z = 0; z < nz; ++z) for (int yl = 0; yl < rows[r]; ++yl) for (int x = 0; x < nx; ++x) {
                size_t k = off[r] + ((size_t)z * rows[r] + yl) * nx + x, gi = IDX(x, y0s[r] + yl, z, nx, ny);
                eC = std::max(eC, (double)fabsf(allC[k] - refC[gi])); eU = std::max(eU, (double)fabsf(allU[k] - refU[gi]));
                mC = std::max(mC, (double)refC[gi]);
            }
        bool pass = eC <= 1e-3 * (mC > 1 ? mC : 1) && eU <= 1e-5;
        printf("{\"backend\": \"%s\", \"ranks\": %d, \"nx\": %d, \"ny\": %d, \"nz\": %d, \"steps\": %d, "
               "\"total_ms\": %.3f, \"compute_ms\": %.3f, \"comm_ms\": %.3f, \"max_err_C\": %.3e, \"max_err_u\": %.3e, \"peak_C\": %.1f, \"pass\": %s}\n",
               BACKEND_NAME, R.P, nx, ny, nz, steps, total / steps * 1e3, comp_max / steps * 1e3, comm_max / steps * 1e3,
               eC, eU, mC, pass ? "true" : "false");
    }
    MPI_Finalize();
    return 0;
}
