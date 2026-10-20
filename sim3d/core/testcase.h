// testcase.h — one fixed, repeatable scenario for benchmarks and for checking MPI against a single run.
// A valley with two hills and a grid of buildings, wind from the south-west, a 10 kg/s styrene leak.
#pragma once
#include "kernels.cuh"
#include <vector>
#include <cmath>

struct TestCase {
    int nx, ny, nz;
    std::vector<uint8_t> solid;          // nz*ny*nx
    std::vector<int> ground, breath;     // ny*nx
    FlowParams F; GasParams G; GasSource src;
};

inline TestCase make_testcase(int nx, int ny, int nz) {
    TestCase T; T.nx = nx; T.ny = ny; T.nz = nz;
    const float dx = 10.0f;
    T.solid.assign((size_t)nx * ny * nz, 0); T.ground.assign((size_t)nx * ny, 0); T.breath.assign((size_t)nx * ny, 0);
    for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
        float X = (float)x / nx, Y = (float)y / ny;
        float terr = 90 * expf(-((X - .15f) * (X - .15f) + (Y - .8f) * (Y - .8f)) / .03f)
                   + 70 * expf(-((X - .85f) * (X - .85f) + (Y - .75f) * (Y - .75f)) / .02f) + 25 * Y;
        float bld = ((x % 9) < 3 && (y % 9) < 3 && terr < 40 && std::abs(x - nx / 2) + std::abs(y - ny / 3) > 6) ? 9.0f : 0.0f;
        int g = (int)ceilf(terr / dx - 0.5f); if (g > nz - 2) g = nz - 2; if (g < 0) g = 0;
        int b = (int)ceilf((terr + bld) / dx - 0.5f); if (b > nz - 2) b = nz - 2; if (b < 0) b = 0;
        T.ground[y * nx + x] = g; T.breath[y * nx + x] = b;
        for (int z = 0; z < nz - 1; ++z) if ((z + 0.5f) * dx < terr + bld) T.solid[IDX(x, y, z, nx, ny)] = 1;
    }
    float dt = 0.33f, beta = 104.15f / 28.97f - 1.0f;
    FlowParams& F = T.F;
    F.nx = nx; F.ny = ny; F.nz = nz; F.gy0 = 0; F.gny = ny;
    F.tau0 = 0.506f; F.cs2 = 0.17f * 0.17f; F.u10 = 2.0f * dt / dx; F.p_exp = 0.3f;
    F.dirx = 0.7071f; F.diry = 0.7071f; F.ref_cells = 1.0f; F.ucap_profile = 0.16f;
    F.sponge_w = nx * 0.06f > 6 ? nx * 0.06f : 6; F.sponge_max = 0.08f; F.g_beta = 9.81f * dt * dt / dx * beta; F.ucap = 0.22f;
    GasParams& G = T.G;
    G.nx = nx; G.ny = ny; G.nz = nz; G.gy0 = 0; G.gny = ny;
    G.sc_inv = 1 / 0.7f; G.kbg_h = 0.4f * dt / (dx * dx); G.kbg_v = 0.05f * dt / (dx * dx); G.kv_factor = 0.15f;
    G.kmax = 0.07f; G.umax = 0.2f; G.decay = 0.0f; G.nsrc = 1;
    int sx = nx / 2, sy = ny / 3;
    float rho_v = 101325.0f * 0.10415f / (8.314f * 300.0f);
    T.src = GasSource{sx, sy, T.breath[sy * nx + sx] + 1, 10.0f / rho_v * dt / (dx * dx * dx) * 1e6f};
    return T;
}
