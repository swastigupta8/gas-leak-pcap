/*
 * gas2d.h - the 2D advection-diffusion model shared by every implementation.
 *
 * Equation (column-averaged concentration C in mg/m^3, uniform wind (u, v), eddy diffusivity D):
 *
 *     dC/dt + u dC/dx + v dC/dy = D (d2C/dx2 + d2C/dy2) + S
 *
 * Discretisation on a square grid of spacing h, explicit in time (forward Euler):
 *   - diffusion: second-order central differences
 *   - advection: first-order upwind (always uses the cell the wind is coming FROM)
 *
 * Both terms only touch the four direct neighbours, so the whole update collapses into a
 * 5-point weighted stencil:
 *
 *     C'(i,j) = w0*C(i,j) + wE*C(i+1,j) + wW*C(i-1,j) + wN*C(i,j+1) + wS*C(i,j-1)
 *
 * with weights fixed for a given (h, dt, D, u, v). Every weight is >= 0 exactly when the time
 * step satisfies the stability limit in g2_dt_max(), which also guarantees concentrations never
 * go negative. See docs/model.md for the derivation.
 *
 * Memory layout: every field is stored with a one-cell "ghost" ring around it, so an array for an
 * nx x ny grid is (nx+2) x (ny+2) floats, row-major, rows running south to north. Interior cells
 * are i = 1..nx, j = 1..ny. Ghost cells hold neighbour data in MPI runs and stay 0 at the outer
 * edge of the map (clean air flowing in, gas leaving the domain).
 */
#ifndef GAS2D_H
#define GAS2D_H

#include <stddef.h>

/* The same inline functions are compiled for the CPU and, inside .cu files, for the GPU. */
#if defined(__CUDACC__)
#define G2_HD __host__ __device__
#else
#define G2_HD
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Stencil weights: centre, east, west, north, south. */
typedef struct {
    float w0, we, ww, wn, ws;
} g2_weights;

/* Physical parameters of one run. */
typedef struct {
    int nx, ny;   /* interior grid size (cells)                    */
    float h;      /* cell size (m)                                  */
    float D;      /* eddy diffusivity (m^2/s)                       */
    float u, v;   /* wind velocity (m/s); +u = towards east, +v = towards north */
    float dt;     /* time step (s)                                  */
} g2_params;

/* Index of padded cell (i, j) in an array whose rows are ld floats long. */
static inline G2_HD size_t g2_idx(int i, int j, int ld) { return (size_t)j * (size_t)ld + (size_t)i; }

/*
 * The 5-point update. The bracketing fixes the order of the floating-point additions, so the
 * serial, OpenMP and MPI versions produce bit-identical fields (the build also disables automatic
 * fused multiply-add for the same reason). The GPU is allowed to use FMA and is checked against a
 * tolerance instead.
 */
static inline G2_HD float g2_stencil(float c, float e, float w, float n, float s, g2_weights k) {
    return k.w0 * c + (k.we * e + k.ww * w) + (k.wn * n + k.ws * s);
}

/* Largest stable time step: dt * (4D/h^2 + (|u|+|v|)/h) <= 1. */
float g2_dt_max(float h, float D, float u, float v);

/* Fill weights for p. Returns 0, or -1 (and leaves *k untouched) if p->dt is unstable. */
int g2_make_weights(const g2_params *p, g2_weights *k);

/* Zeroed padded field for an nx x ny grid; NULL if out of memory. Free with g2_free. */
float *g2_alloc(int nx, int ny);
void g2_free(float *p);

/* One step over the padded rectangle [i0, i1) x [j0, j1) (padded coordinates). */
void g2_step_rect(const float *in, float *out, int ld, int i0, int i1, int j0, int j1, g2_weights k);

/* One full step over all interior cells: single thread, and OpenMP-parallel. */
void g2_step_serial(const float *in, float *out, int nx, int ny, g2_weights k);
void g2_step_omp(const float *in, float *out, int nx, int ny, g2_weights k);

/*
 * Analytic solution for an instantaneous release of M mg per metre of mixing height at (x0, y0),
 * evaluated t seconds later (wind drift included):
 *     C = M / (4 pi D t) * exp(-((x - x0 - u t)^2 + (y - y0 - v t)^2) / (4 D t))
 * Writes the value at every interior cell centre of a local block whose first interior cell is
 * global cell (gx0, gy0). Used to start the validation test and to check its result.
 */
void g2_gaussian_fill(float *C, int ld, int lx, int ly, int gx0, int gy0, const g2_params *p,
                      double x0, double y0, double M, double t);

/* Total mass per metre of mixing height (mg/m): sum of interior cells times h^2. */
double g2_mass(const float *C, int nx, int ny, float h);

/* Wall-clock time in seconds (monotonic). */
double g2_now(void);

#ifdef __cplusplus
}
#endif

#endif /* GAS2D_H */
