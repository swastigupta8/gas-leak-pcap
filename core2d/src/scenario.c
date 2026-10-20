/*
 * scenario.c - option parsing, initial conditions and JSON reporting (see scenario.h).
 */
#include "scenario.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int need(int i, int argc, const char *name) {
    if (i + 1 >= argc) {
        fprintf(stderr, "missing value after %s\n", name);
        return 0;
    }
    return 1;
}

int g2_parse(int argc, char **argv, g2_config *c, const char *default_variant) {
    memset(c, 0, sizeof *c);
    snprintf(c->variant, sizeof c->variant, "%s", default_variant);
    c->nx = c->ny = 1024;
    c->steps = 200;
    c->h = 1.0f;
    c->D = 10.0f;
    c->u = 3.0f;
    c->v = 1.0f;
    c->dt = 0.0f; /* 0 = choose automatically */

    int src_given = 0;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
#define ARG(name) (strcmp(a, name) == 0 && need(i, argc, name))
        if (ARG("--nx")) c->nx = atoi(argv[++i]);
        else if (ARG("--ny")) c->ny = atoi(argv[++i]);
        else if (ARG("--steps")) c->steps = atoi(argv[++i]);
        else if (ARG("--h")) c->h = (float)atof(argv[++i]);
        else if (ARG("--D")) c->D = (float)atof(argv[++i]);
        else if (ARG("--u")) c->u = (float)atof(argv[++i]);
        else if (ARG("--v")) c->v = (float)atof(argv[++i]);
        else if (ARG("--dt")) c->dt = (float)atof(argv[++i]);
        else if (ARG("--threads")) c->threads = atoi(argv[++i]);
        else if (ARG("--out")) c->out = argv[++i];
        else if (ARG("--variant")) snprintf(c->variant, sizeof c->variant, "%s", argv[++i]);
        else if (ARG("--source")) {
            if (sscanf(argv[++i], "%d,%d,%f", &c->src_i, &c->src_j, &c->src_rate) != 3) {
                fprintf(stderr, "--source expects I,J,RATE\n");
                return -1;
            }
            src_given = 1;
        } else if (ARG("--gauss")) {
            if (sscanf(argv[++i], "%lf,%lf,%lf,%lf", &c->gx, &c->gy, &c->gM, &c->gt0) != 4) {
                fprintf(stderr, "--gauss expects X,Y,M,T0\n");
                return -1;
            }
            c->has_gauss = 1;
        } else if (strcmp(a, "--overlap") == 0) c->overlap = 1;
        else {
            fprintf(stderr, "unknown or incomplete option: %s\n", a);
            return -1;
        }
#undef ARG
    }

    if (c->nx < 1 || c->ny < 1 || c->steps < 0 || c->h <= 0.0f || c->D < 0.0f) {
        fprintf(stderr, "invalid grid or physical parameters\n");
        return -1;
    }
    /* A puff test has no leak unless one was asked for explicitly; otherwise default to a leak
     * in the middle of the map. */
    if (src_given) c->has_source = 1;
    else if (!c->has_gauss) {
        c->has_source = 1;
        c->src_i = c->nx / 2;
        c->src_j = c->ny / 2;
        c->src_rate = 1e4f;
    }
    if (c->has_source && (c->src_i < 0 || c->src_i >= c->nx || c->src_j < 0 || c->src_j >= c->ny)) {
        fprintf(stderr, "source cell is outside the grid\n");
        return -1;
    }
    return 0;
}

int g2_setup(const g2_config *c, g2_params *p, g2_weights *k) {
    p->nx = c->nx;
    p->ny = c->ny;
    p->h = c->h;
    p->D = c->D;
    p->u = c->u;
    p->v = c->v;
    float lim = g2_dt_max(c->h, c->D, c->u, c->v);
    /* 0.9 x the limit: right at the limit the centre weight is exactly 0, which is stable but
     * gives the least smooth results; a small margin is standard practice. */
    p->dt = c->dt > 0.0f ? c->dt : 0.9f * lim;
    if (g2_make_weights(p, k) != 0) {
        fprintf(stderr, "unstable: dt = %g s but the limit for h=%g D=%g u=%g v=%g is %g s\n", p->dt, p->h,
                p->D, p->u, p->v, lim);
        return -1;
    }
    return 0;
}

void g2_initial_block(const g2_config *c, const g2_params *p, float *C, int ld, int lx, int ly, int gx0,
                      int gy0) {
    if (c->has_gauss) g2_gaussian_fill(C, ld, lx, ly, gx0, gy0, p, c->gx, c->gy, c->gM, c->gt0);
    /* otherwise the field starts as clean air (already zero) */
}

float g2_source_per_step(const g2_config *c, const g2_params *p) {
    return c->has_source ? c->src_rate * p->dt : 0.0f;
}

int g2_write_field(const char *path, const float *C, int nx, int ny) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int ok = 1;
    for (int j = 1; j <= ny && ok; ++j) ok = fwrite(C + g2_idx(1, j, nx + 2), sizeof(float), (size_t)nx, f) == (size_t)nx;
    return (fclose(f) == 0 && ok) ? 0 : -1;
}

void g2_report(const g2_config *c, const g2_params *p, double ms_per_step, const char *extra,
               const float *global) {
    const double cells = (double)c->nx * c->ny;
    /* Throughput in million lattice-cell updates per second (MLUPS), the standard stencil metric. */
    const double mlups = ms_per_step > 0 ? cells / (ms_per_step * 1e3) : 0.0;
    /* Minimum memory traffic per step: read every cell once and write it once (4 bytes each).
     * Neighbour re-reads should come from cache; if they do not, real traffic is higher. */
    const double gbps = ms_per_step > 0 ? 8.0 * cells / (ms_per_step * 1e6) : 0.0;
    /* How many simulated seconds pass per second of computing. Above 1 = faster than real time. */
    const double realtime = ms_per_step > 0 ? (p->dt * 1e3) / ms_per_step : 0.0;

    printf("{\"variant\":\"%s\",\"nx\":%d,\"ny\":%d,\"steps\":%d,\"dt\":%.6g,\"ms_per_step\":%.6g,"
           "\"mlups\":%.6g,\"gbps\":%.6g,\"realtime_factor\":%.6g",
           c->variant, c->nx, c->ny, c->steps, p->dt, ms_per_step, mlups, gbps, realtime);
    if (global) printf(",\"mass\":%.10g", g2_mass(global, c->nx, c->ny, c->h));
    if (extra && *extra) printf(",%s", extra);
    printf("}\n");
    fflush(stdout);

    if (global && c->out && g2_write_field(c->out, global, c->nx, c->ny) != 0)
        fprintf(stderr, "could not write %s\n", c->out);
}
