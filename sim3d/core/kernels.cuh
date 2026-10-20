// kernels.cuh — all the physics, one GPU thread per grid box.
//
// Grid: NX x NY x NZ boxes of dx metres. x = east, y = north, z = up.
// Memory index of box (x,y,z):  i = (z*NY + y)*NX + x   (x varies fastest -> neighbouring threads
// read neighbouring memory, which the GPU can combine into one fast transaction: "coalescing").
//
// Two coupled models run every time step:
//   1. AIR  — Lattice Boltzmann, D3Q19 (19 packet directions per box), with:
//             * Smagorinsky turbulence model (LES): extra mixing where the flow is sheared
//             * buoyancy: boxes containing heavy styrene vapour are pulled down by gravity
//             * terrain + buildings: packets bounce back off solid boxes
//             * "sponge" edges: near the map edges the air is gently pushed toward the
//               measured wind profile, so wind can blow in from any direction
//   2. GAS  — finite-volume advection–diffusion of styrene concentration (ppm), carried by the
//             air velocity from step 1, mixed by the turbulence from step 1.
#pragma once
#include "compat.h"

// ---------------------------------------------------------------------------------------------
// D3Q19 lattice: 19 directions (rest, 6 faces, 12 edges) and their weights
// ---------------------------------------------------------------------------------------------
__constant__ int EX[19]  = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
__constant__ int EY[19]  = {0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1};
__constant__ int EZ[19]  = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1};
__constant__ int OPP[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7,10, 9,12,11,14,13,16,15,18,17};
__constant__ float WQ[19] = {1.f/3,
    1.f/18, 1.f/18, 1.f/18, 1.f/18, 1.f/18, 1.f/18,
    1.f/36, 1.f/36, 1.f/36, 1.f/36, 1.f/36, 1.f/36, 1.f/36, 1.f/36, 1.f/36, 1.f/36, 1.f/36, 1.f/36};

#define MAX_SRC 32
struct GasSource { int x, y, z; float rate; };          // rate = ppm added to that box per step
__constant__ GasSource c_src[MAX_SRC];

// Everything the air model needs, in lattice units (1 box = 1, 1 step = 1).
struct FlowParams {
    int nx, ny, nz;
    int gy0, gny;        // MPI: global row of local row 0, and global number of rows (single run: 0, ny)
    float tau0;          // base relaxation time (sets the base viscosity)
    float cs2;           // Smagorinsky constant squared
    float u10;           // wind speed at 10 m above ground (lattice units)
    float p_exp;         // wind profile exponent: u(z) = u10 * (z/10m)^p  (bigger = more stable night air)
    float dirx, diry;    // unit vector of where the wind blows TO
    float ref_cells;     // how many boxes make 10 m
    float ucap_profile;  // fastest wind allowed in the profile (keeps LBM stable)
    float sponge_w;      // sponge width at the edges (boxes)
    float sponge_max;    // strongest sponge pull per step
    float g_beta;        // gravity * (M_gas/M_air - 1), lattice units: buoyancy per unit mole fraction
    float ucap;          // absolute safety cap on lattice speed
};

struct GasParams {
    int nx, ny, nz;
    int gy0, gny;        // MPI: as in FlowParams
    float sc_inv;        // 1 / turbulent Schmidt number (how turbulence mixes gas vs momentum)
    float kbg_h, kbg_v;  // background mixing, horizontal and vertical (lattice units)
    float kv_factor;     // stability: <1 suppresses vertical turbulent mixing (stable night air)
    float kmax;          // cap on mixing (explicit-scheme stability)
    float umax;          // cap on velocity used for gas (explicit-scheme stability)
    float decay;         // fraction lost per step (chemical breakdown / washout)
    int nsrc;
};

HD size_t IDX(int x, int y, int z, int nx, int ny) { return ((size_t)z * ny + y) * nx + x; }

DEV float feq(int q, float rho, float ux, float uy, float uz) {
    float eu = EX[q] * ux + EY[q] * uy + EZ[q] * uz;
    float usq = ux * ux + uy * uy + uz * uz;
    return WQ[q] * rho * (1.0f + 3.0f * eu + 4.5f * eu * eu - 1.5f * usq);
}

// The measured wind at this box: power-law profile above the local ground, zero below it.
HD void target_wind(const FlowParams& P, int z, int ground, float& u, float& v) {
    float h = (z - ground + 0.5f) / P.ref_cells;          // height above ground in units of 10 m
    float s = 0.0f;
    if (z >= ground) {
        if (h < 0.05f) h = 0.05f;
        s = P.u10 * powf(h, P.p_exp);
        if (s > P.ucap_profile) s = P.ucap_profile;
    }
    u = s * P.dirx; v = s * P.diry;
}

HD float sponge_strength(const FlowParams& P, int x, int y, int z) {
    int gy = y + P.gy0;                                    // global row (MPI-safe)
    int d = x; if (P.nx - 1 - x < d) d = P.nx - 1 - x;
    if (gy < d) d = gy; if (P.gny - 1 - gy < d) d = P.gny - 1 - gy;
    float a = 1.0f - d / P.sponge_w;                      // lateral edges
    float b = 1.0f - (P.nz - 1 - z) / (0.5f * P.sponge_w); // top
    float s = a > b ? a : b;
    if (s <= 0.0f) return 0.0f;
    return P.sponge_max * s * s;
}

// ---------------------------------------------------------------------------------------------
// Set the air to the measured wind profile everywhere (start of a run).
// ---------------------------------------------------------------------------------------------
__global__ void init_flow(float* f, const uint8_t* solid, const int* ground, FlowParams P) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    if (x >= P.nx || y >= P.ny || z >= P.nz) return;
    size_t N = (size_t)P.nx * P.ny * P.nz, i = IDX(x, y, z, P.nx, P.ny);
    float u, v; target_wind(P, z, ground[y * P.nx + x], u, v);
    for (int q = 0; q < 19; ++q) f[q * N + i] = solid[i] ? 0.0f : feq(q, 1.0f, u, v, 0.0f);
}

// ---------------------------------------------------------------------------------------------
// AIR: one Lattice Boltzmann step ("pull" scheme: each box gathers the packets streaming into it,
// then collides them). Reads fin, writes fout — like Stage 1, never read and write the same grid.
// ---------------------------------------------------------------------------------------------
__global__ void lbm_step(const float* __restrict__ fin, float* __restrict__ fout,
                         const uint8_t* __restrict__ solid, const int* __restrict__ ground,
                         const float* __restrict__ C,
                         float* __restrict__ ux_out, float* __restrict__ uy_out,
                         float* __restrict__ uz_out, float* __restrict__ nut_out, FlowParams P) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    if (x >= P.nx || y >= P.ny || z >= P.nz) return;
    size_t N = (size_t)P.nx * P.ny * P.nz, i = IDX(x, y, z, P.nx, P.ny);
    if (solid[i]) { ux_out[i] = uy_out[i] = uz_out[i] = nut_out[i] = 0.0f; return; }

    int gnd = ground[y * P.nx + x];
    float tu, tv; target_wind(P, z, gnd, tu, tv);

    // 1. STREAM (pull): packet q arriving here left box (x,y,z) - e_q last step
    float f[19];
    for (int q = 0; q < 19; ++q) {
        int sx = x - EX[q], sy = y - EY[q], sz = z - EZ[q];
        if (sz < 0) {
            f[q] = fin[OPP[q] * N + i];                        // floor of the world: bounce back
        } else if (sx < 0 || sx >= P.nx || sy < 0 || sy >= P.ny || sz >= P.nz ||
                   sy + P.gy0 < 0 || sy + P.gy0 >= P.gny) {
            f[q] = feq(q, 1.0f, tu, tv, 0.0f);                 // from outside the map: measured wind
        } else {
            size_t j = IDX(sx, sy, sz, P.nx, P.ny);
            f[q] = solid[j] ? fin[OPP[q] * N + i]              // came from a wall: bounced back
                            : fin[q * N + j];
        }
    }

    // 2. MEASURE density and momentum
    float rho = 0, mx = 0, my = 0, mz = 0;
    for (int q = 0; q < 19; ++q) { rho += f[q]; mx += f[q] * EX[q]; my += f[q] * EY[q]; mz += f[q] * EZ[q]; }

    // 3. BUOYANCY: heavy vapour makes the mixture denser -> pulled down
    float X = C[i] * 1e-6f;                                    // mole fraction (ppm -> fraction)
    if (X > 0.5f) X = 0.5f;
    float Fz = -P.g_beta * X * rho;

    float ux = mx / rho, uy = my / rho, uz = (mz + 0.5f * Fz) / rho;

    // 4. TURBULENCE (Smagorinsky LES): find the local shear from the non-equilibrium packets
    float pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
    for (int q = 0; q < 19; ++q) {
        float fn = f[q] - feq(q, rho, ux, uy, uz);
        pxx += fn * EX[q] * EX[q]; pyy += fn * EY[q] * EY[q]; pzz += fn * EZ[q] * EZ[q];
        pxy += fn * EX[q] * EY[q]; pxz += fn * EX[q] * EZ[q]; pyz += fn * EY[q] * EZ[q];
    }
    float Qn = sqrtf(2.0f * (pxx * pxx + pyy * pyy + pzz * pzz + 2.0f * (pxy * pxy + pxz * pxz + pyz * pyz)));
    float tau = 0.5f * (P.tau0 + sqrtf(P.tau0 * P.tau0 + 18.0f * 1.41421356f * P.cs2 * Qn / rho));

    // 5. COLLIDE toward equilibrium (velocity shifted by the force = Shan–Chen forcing)
    float ex_ = ux, ey_ = uy, ez_ = (mz + tau * Fz) / rho;
    float sp = sqrtf(ex_ * ex_ + ey_ * ey_ + ez_ * ez_);
    if (sp > P.ucap) { float s = P.ucap / sp; ex_ *= s; ey_ *= s; ez_ *= s; }
    float sg = sponge_strength(P, x, y, z);
    for (int q = 0; q < 19; ++q) {
        float fq = f[q] - (f[q] - feq(q, rho, ex_, ey_, ez_)) / tau;
        if (sg > 0.0f) fq = (1.0f - sg) * fq + sg * feq(q, 1.0f, tu, tv, 0.0f);
        fout[q * N + i] = fq;
    }
    ux_out[i] = ux; uy_out[i] = uy; uz_out[i] = uz;
    nut_out[i] = (tau - 0.5f) / 3.0f;                          // turbulent viscosity
}

// ---------------------------------------------------------------------------------------------
// GAS: flux through one face of a box (shared by the naive and shared-memory kernels, so both
// give bit-identical answers). `un` = velocity across the face, positive = leaving my box.
// Upwind advection + diffusion. The neighbour computes the same flux with the opposite sign,
// so gas is never created or destroyed at faces (mass is conserved exactly).
// ---------------------------------------------------------------------------------------------
HD float face_out(float c, float cn, float un, float K) {
    float adv = (un > 0.0f) ? un * c : un * cn;
    return adv + K * (c - cn);
}
HD float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
HD float mix_h(const GasParams& G, float nut) { return clampf(nut * G.sc_inv + G.kbg_h, 0.0f, G.kmax); }
HD float mix_v(const GasParams& G, float nut) { return clampf(nut * G.sc_inv * G.kv_factor + G.kbg_v, 0.0f, G.kmax); }

// All the gas physics for one box, given its 6 neighbours' concentrations.
// solid neighbour -> face closed; outside the map -> clean air outside, gas can leave.
DEV float gas_update(const GasParams& G, int x, int y, int z, size_t i, float c,
                    float cE, float cW, float cN, float cS, float cU, float cD,
                    const uint8_t* solid, const float* ux, const float* uy, const float* uz,
                    const float* nut) {
    size_t sx = 1, sy = (size_t)G.nx, sz = (size_t)G.nx * G.ny;
    float Kh = mix_h(G, nut[i]), Kv = mix_v(G, nut[i]);
    float ui = clampf(ux[i], -G.umax, G.umax), vi = clampf(uy[i], -G.umax, G.umax), wi = clampf(uz[i], -G.umax, G.umax);
    float out = 0.0f;
    // east (+x)
    if (x + 1 < G.nx) { size_t j = i + sx; if (!solid[j]) out += face_out(c, cE, 0.5f * (ui + clampf(ux[j], -G.umax, G.umax)), 0.5f * (Kh + mix_h(G, nut[j]))); }
    else out += face_out(c, 0.0f, ui, Kh);
    // west (-x)
    if (x > 0) { size_t j = i - sx; if (!solid[j]) out += face_out(c, cW, -0.5f * (ui + clampf(ux[j], -G.umax, G.umax)), 0.5f * (Kh + mix_h(G, nut[j]))); }
    else out += face_out(c, 0.0f, -ui, Kh);
    // north (+y)   (open edge only at the edge of the WHOLE map, not at an MPI strip boundary)
    if (y + 1 < G.ny && y + G.gy0 + 1 < G.gny) { size_t j = i + sy; if (!solid[j]) out += face_out(c, cN, 0.5f * (vi + clampf(uy[j], -G.umax, G.umax)), 0.5f * (Kh + mix_h(G, nut[j]))); }
    else out += face_out(c, 0.0f, vi, Kh);
    // south (-y)
    if (y > 0 && y + G.gy0 > 0) { size_t j = i - sy; if (!solid[j]) out += face_out(c, cS, -0.5f * (vi + clampf(uy[j], -G.umax, G.umax)), 0.5f * (Kh + mix_h(G, nut[j]))); }
    else out += face_out(c, 0.0f, -vi, Kh);
    // up (+z)
    if (z + 1 < G.nz) { size_t j = i + sz; if (!solid[j]) out += face_out(c, cU, 0.5f * (wi + clampf(uz[j], -G.umax, G.umax)), 0.5f * (Kv + mix_v(G, nut[j]))); }
    else out += face_out(c, 0.0f, wi, Kv);
    // down (-z): the floor of the world is closed
    if (z > 0) { size_t j = i - sz; if (!solid[j]) out += face_out(c, cD, -0.5f * (wi + clampf(uz[j], -G.umax, G.umax)), 0.5f * (Kv + mix_v(G, nut[j]))); }

    float cn = (c - out) * (1.0f - G.decay);
    for (int s = 0; s < G.nsrc; ++s)
        if (c_src[s].x == x && c_src[s].y == y + G.gy0 && c_src[s].z == z) cn += c_src[s].rate;
    return cn > 0.0f ? cn : 0.0f;
}

// Version A — NAIVE: every neighbour value read straight from global GPU memory.
__global__ void gas_step_naive(const float* __restrict__ Cin, float* __restrict__ Cout,
                               const uint8_t* __restrict__ solid, const float* __restrict__ ux,
                               const float* __restrict__ uy, const float* __restrict__ uz,
                               const float* __restrict__ nut, GasParams G) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    if (x >= G.nx || y >= G.ny || z >= G.nz) return;
    size_t i = IDX(x, y, z, G.nx, G.ny), sy = G.nx, sz = (size_t)G.nx * G.ny;
    if (solid[i]) { Cout[i] = 0.0f; return; }
    float c = Cin[i];
    float cE = x + 1 < G.nx ? Cin[i + 1] : 0.f, cW = x > 0 ? Cin[i - 1] : 0.f;
    float cN = y + 1 < G.ny ? Cin[i + sy] : 0.f, cS = y > 0 ? Cin[i - sy] : 0.f;
    float cU = z + 1 < G.nz ? Cin[i + sz] : 0.f, cD = z > 0 ? Cin[i - sz] : 0.f;
    Cout[i] = gas_update(G, x, y, z, i, c, cE, cW, cN, cS, cU, cD, solid, ux, uy, uz, nut);
}

// Version B — SHARED MEMORY ("2.5D blocking"). A block of 32x8 threads owns a 32x8 column of
// boxes and walks UP through it one level at a time:
//   * the current level's 32x8 tile plus a 1-box halo ring is copied into fast shared memory,
//     so the 4 sideways neighbours are read from shared memory instead of global memory;
//   * the box below and above are kept in registers and shifted as the block climbs.
// Each concentration value is fetched from global memory ~1.3 times instead of 7 times.
#define TBX 32
#define TBY 8
__global__ void gas_step_shared(const float* __restrict__ Cin, float* __restrict__ Cout,
                                const uint8_t* __restrict__ solid, const float* __restrict__ ux,
                                const float* __restrict__ uy, const float* __restrict__ uz,
                                const float* __restrict__ nut, GasParams G) {
    __shared__ float tile[TBY + 2][TBX + 2];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * TBX + tx, y = blockIdx.y * TBY + ty;
    bool inside = (x < G.nx && y < G.ny);
    size_t sz = (size_t)G.nx * G.ny;
    size_t col = inside ? (size_t)y * G.nx + x : 0;
    // registers: value below, here, above
    float cD = 0.0f;
    float cC = inside ? Cin[col] : 0.0f;
    float cU = (inside && G.nz > 1) ? Cin[col + sz] : 0.0f;

    for (int z = 0; z < G.nz; ++z) {
        SYNC_THREADS();                                   // everyone done with the previous level
        tile[ty + 1][tx + 1] = cC;
        size_t base = (size_t)z * sz;
        // halo ring (threads on the tile edge also fetch the box just outside the tile)
        if (tx == 0)       tile[ty + 1][0]       = (inside && x > 0)        ? Cin[base + col - 1]    : 0.0f;
        if (tx == TBX - 1) tile[ty + 1][TBX + 1] = (y < G.ny && x + 1 < G.nx) ? Cin[base + (size_t)y * G.nx + x + 1] : 0.0f;
        if (ty == 0)       tile[0][tx + 1]       = (x < G.nx && y > 0)      ? Cin[base + (size_t)(y - 1) * G.nx + x] : 0.0f;
        if (ty == TBY - 1) tile[TBY + 1][tx + 1] = (x < G.nx && y + 1 < G.ny) ? Cin[base + (size_t)(y + 1) * G.nx + x] : 0.0f;
        SYNC_THREADS();                                   // tile complete

        if (inside) {
            size_t i = base + col;
            if (solid[i]) Cout[i] = 0.0f;
            else Cout[i] = gas_update(G, x, y, z, i, cC,
                                      tile[ty + 1][tx + 2], tile[ty + 1][tx], tile[ty + 2][tx + 1], tile[ty][tx + 1],
                                      cU, cD, solid, ux, uy, uz, nut);
        }
        // climb one level: shift the registers
        cD = cC; cC = cU;
        cU = (inside && z + 2 < G.nz) ? Cin[base + 2 * sz + col] : 0.0f;
    }
}

// ---------------------------------------------------------------------------------------------
// OUTPUT kernels: shrink the 3D data on the GPU before it travels to the browser.
// ---------------------------------------------------------------------------------------------
// Concentration at breathing height (first air box above ground or roof) for every column.
__global__ void ground_level(const float* __restrict__ C, const int* __restrict__ breath,
                             float* __restrict__ out, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    out[y * nx + x] = C[IDX(x, y, breath[y * nx + x], nx, ny)];
}

// ppm -> one byte on a log scale: 0 = below 0.1 ppm, 255 = 10,000 ppm or more.
HD uint8_t ppm_to_byte(float ppm) {
    if (ppm < 0.1f) return 0;
    float t = (log10f(ppm) + 1.0f) / 5.0f;               // 0.1..1e4 ppm -> 0..1
    int b = 1 + (int)(t * 254.0f);
    return (uint8_t)(b > 255 ? 255 : b);
}

// 3D volume shrunk by (fx, fy, fz), keeping the maximum in each block, as bytes.
__global__ void volume_bytes(const float* __restrict__ C, uint8_t* __restrict__ out,
                             int nx, int ny, int nz, int fx, int fy, int fz, int vx, int vy, int vz) {
    int a = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (a >= vx || b >= vy || c >= vz) return;
    float m = 0.0f;
    for (int dz = 0; dz < fz; ++dz) for (int dy = 0; dy < fy; ++dy) for (int dx = 0; dx < fx; ++dx) {
        int x = a * fx + dx, y = b * fy + dy, z = c * fz + dz;
        if (x < nx && y < ny && z < nz) { float v = C[IDX(x, y, z, nx, ny)]; if (v > m) m = v; }
    }
    out[((size_t)c * vy + b) * vx + a] = ppm_to_byte(m);
}

// Wind at a fixed height above ground, shrunk by f, for the wind particles in the browser.
__global__ void wind_slice(const float* __restrict__ ux, const float* __restrict__ uy,
                           const int* __restrict__ ground, float* __restrict__ out,
                           int nx, int ny, int nz, int f, int zoff, int wx, int wy) {
    int a = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= wx || b >= wy) return;
    int x = a * f + f / 2, y = b * f + f / 2;
    if (x >= nx) x = nx - 1; if (y >= ny) y = ny - 1;
    int z = ground[y * nx + x] + zoff; if (z >= nz) z = nz - 1;
    size_t i = IDX(x, y, z, nx, ny);
    out[2 * (b * wx + a)] = ux[i]; out[2 * (b * wx + a) + 1] = uy[i];
}

// 3D wind for the "Boxes" view, shrunk by (fx, fy, fz): one sample per block, in 0.1 m/s steps,
// stored as int8 (east, north, up planes one after another). Solid boxes give zero wind.
__global__ void wind_volume(const float* __restrict__ ux, const float* __restrict__ uy,
                            const float* __restrict__ uz, const uint8_t* __restrict__ solid,
                            int8_t* __restrict__ out, int nx, int ny, int nz, int fx, int fy, int fz,
                            int vx, int vy, int vz, float scale) {
    int a = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.z;
    if (a >= vx || b >= vy || c >= vz) return;
    int x = a * fx + fx / 2, y = b * fy + fy / 2, z = c * fz + fz / 2;
    if (x >= nx) x = nx - 1; if (y >= ny) y = ny - 1; if (z >= nz) z = nz - 1;
    size_t i = IDX(x, y, z, nx, ny), plane = (size_t)vx * vy * vz, o = ((size_t)c * vy + b) * vx + a;
    float v[3] = {ux[i], uy[i], uz[i]};
    for (int k = 0; k < 3; ++k) {
        float q = solid[i] ? 0.0f : v[k] * scale;
        q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
        out[k * plane + o] = (int8_t)lrintf(q);
    }
}
