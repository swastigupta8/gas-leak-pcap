/*
 * lib_cuda.cu - live-simulation library on the GPU (see gas2d_lib.h). Uses the coarsened kernel,
 * which is expected to be the fastest; the benchmark (gas2d_cuda) measures all three.
 */
#include "cuda_backend.h"
#include "gas2d_lib.h"

#include <cstdio>
#include <cstdlib>

struct g2_sim {
    g2_params p;
    g2_weights k;
    g2_gpu g;
    long long src;
    float s;
};

extern "C" {

g2_sim *g2s_create(int nx, int ny) {
    g2_sim *s = (g2_sim *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->p.nx = nx;
    s->p.ny = ny;
    s->src = -1;
    if (g2_gpu_create(&s->g, nx, ny) != 0) {
        free(s);
        return NULL;
    }
    g2s_set_params(s, 1.0f, 10.0f, 0.0f, 0.0f, 0.0f);
    return s;
}

void g2s_destroy(g2_sim *s) {
    if (!s) return;
    g2_gpu_destroy(&s->g);
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

void g2s_clear(g2_sim *s) { g2_gpu_zero(&s->g); }

double g2s_step(g2_sim *s, int n) { return g2_gpu_run(&s->g, G2_COARSE, n, s->k, s->src, s->s); }

void g2s_frame(g2_sim *s, float *out, int fx, int fy) { g2_gpu_frame(&s->g, out, fx, fy); }

const char *g2s_backend(void) {
    static char name[160];
    if (!name[0]) {
        char dev[128];
        double peak;
        g2_gpu_info(dev, sizeof dev, &peak);
        snprintf(name, sizeof name, "cuda:%s", dev);
    }
    return name;
}

} /* extern "C" */
