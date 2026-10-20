/*
 * common.c - model set-up shared by every variant: stability limit, stencil weights, field
 * allocation, the serial stencil sweep, the analytic solution and small utilities.
 */
#include "gas2d.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

float g2_dt_max(float h, float D, float u, float v) {
    /* The centre weight is 1 - dt*(4D/h^2 + (|u|+|v|)/h). It must stay >= 0; otherwise the update
     * over-shoots, concentrations oscillate and go negative, and the run blows up. */
    double rate = 4.0 * D / ((double)h * h) + (fabs((double)u) + fabs((double)v)) / h;
    return rate > 0.0 ? (float)(1.0 / rate) : INFINITY;
}

int g2_make_weights(const g2_params *p, g2_weights *k) {
    if (!(p->dt > 0.0f) || p->dt > g2_dt_max(p->h, p->D, p->u, p->v)) return -1;

    const double h = p->h, dt = p->dt;
    const double dif = dt * p->D / (h * h);       /* diffusion share, same for all 4 neighbours   */
    const double up = dt * fmax(p->u, 0.0) / h;   /* wind blowing east  -> take from the west cell  */
    const double um = dt * fmax(-p->u, 0.0) / h;  /* wind blowing west  -> take from the east cell  */
    const double vp = dt * fmax(p->v, 0.0) / h;   /* wind blowing north -> take from the south cell */
    const double vm = dt * fmax(-p->v, 0.0) / h;  /* wind blowing south -> take from the north cell */

    k->we = (float)(dif + um);
    k->ww = (float)(dif + up);
    k->wn = (float)(dif + vm);
    k->ws = (float)(dif + vp);
    /* Centre weight = 1 - sum of the others, so the weights add up to 1: a uniform field stays
     * uniform and no mass is created or destroyed inside the domain. */
    k->w0 = (float)(1.0 - (4.0 * dif + up + um + vp + vm));
    return 0;
}

float *g2_alloc(int nx, int ny) {
    size_t n = (size_t)(nx + 2) * (size_t)(ny + 2);
    /* calloc: the ghost ring must start (and, at the map edge, stay) at zero. */
    return (float *)calloc(n, sizeof(float));
}

void g2_free(float *p) { free(p); }

void g2_step_rect(const float *restrict in, float *restrict out, int ld, int i0, int i1, int j0, int j1,
                  g2_weights k) {
    for (int j = j0; j < j1; ++j) {
        /* Row pointers keep the inner loop simple enough for the compiler to vectorise it. */
        const float *row = in + (size_t)j * ld;
        const float *north = row + ld;
        const float *south = row - ld;
        float *dst = out + (size_t)j * ld;
        for (int i = i0; i < i1; ++i)
            dst[i] = g2_stencil(row[i], row[i + 1], row[i - 1], north[i], south[i], k);
    }
}

void g2_step_serial(const float *in, float *out, int nx, int ny, g2_weights k) {
    g2_step_rect(in, out, nx + 2, 1, nx + 1, 1, ny + 1, k);
}

void g2_gaussian_fill(float *C, int ld, int lx, int ly, int gx0, int gy0, const g2_params *p, double x0,
                      double y0, double M, double t) {
    const double h = p->h, D = p->D;
    const double cx = x0 + p->u * t, cy = y0 + p->v * t; /* the puff centre drifts with the wind */
    const double amp = M / (4.0 * M_PI * D * t);
    for (int j = 1; j <= ly; ++j) {
        double y = (gy0 + j - 1 + 0.5) * h - cy; /* cell centre, measured from the puff centre */
        for (int i = 1; i <= lx; ++i) {
            double x = (gx0 + i - 1 + 0.5) * h - cx;
            C[g2_idx(i, j, ld)] = (float)(amp * exp(-(x * x + y * y) / (4.0 * D * t)));
        }
    }
}

double g2_mass(const float *C, int nx, int ny, float h) {
    double s = 0.0;
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) s += C[g2_idx(i, j, nx + 2)];
    return s * (double)h * h;
}

double g2_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
