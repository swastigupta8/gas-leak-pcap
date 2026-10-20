/*
 * lib_cpu.c - live-simulation library on the CPU with OpenMP (see gas2d_lib.h).
 */
#include "gas2d_lib.h"
#include "kernels.h"

#include <stdlib.h>
#include <string.h>

struct g2_sim {
    g2_params p;
    g2_weights k;
    float *a, *b;
    long long src;
    float s;
};

g2_sim *g2s_create(int nx, int ny) {
    g2_sim *s = (g2_sim *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->p.nx = nx;
    s->p.ny = ny;
    s->a = g2_alloc(nx, ny);
    s->b = g2_alloc(nx, ny);
    s->src = -1;
    if (!s->a || !s->b) {
        g2s_destroy(s);
        return NULL;
    }
    g2s_set_params(s, 1.0f, 10.0f, 0.0f, 0.0f, 0.0f);
    return s;
}

void g2s_destroy(g2_sim *s) {
    if (!s) return;
    g2_free(s->a);
    g2_free(s->b);
    free(s);
}

float g2s_set_params(g2_sim *s, float h, float D, float u, float v, float dt) {
    g2_params p = s->p;
    p.h = h;
    p.D = D;
    p.u = u;
    p.v = v;
    p.dt = dt > 0.0f ? dt : 0.9f * g2_dt_max(h, D, u, v);
    g2_weights k;
    if (g2_make_weights(&p, &k) != 0) return -1.0f;
    s->p = p;
    s->k = k;
    return p.dt;
}

void g2s_set_source(g2_sim *s, int i, int j, float per_step) {
    if (i < 0 || j < 0 || i >= s->p.nx || j >= s->p.ny) {
        s->src = -1;
        return;
    }
    s->src = (long long)g2_idx(i + 1, j + 1, s->p.nx + 2);
    s->s = per_step;
}

void g2s_clear(g2_sim *s) {
    size_t n = (size_t)(s->p.nx + 2) * (s->p.ny + 2) * sizeof(float);
    memset(s->a, 0, n);
    memset(s->b, 0, n);
}

double g2s_step(g2_sim *s, int n) {
    double t0 = g2_now();
    for (int i = 0; i < n; ++i) {
        g2_step_omp(s->a, s->b, s->p.nx, s->p.ny, s->k);
        if (s->src >= 0) s->b[s->src] += s->s;
        float *t = s->a; s->a = s->b; s->b = t;
    }
    return (g2_now() - t0) * 1e3;
}

void g2s_frame(g2_sim *s, float *out, int fx, int fy) {
#pragma omp parallel for schedule(static)
    for (int oy = 0; oy < fy; ++oy)
        for (int ox = 0; ox < fx; ++ox) out[(size_t)oy * fx + ox] = g2_frame_pixel(s->a, s->p.nx, s->p.ny, fx, fy, ox, oy);
}

const char *g2s_backend(void) { return "cpu-openmp"; }
