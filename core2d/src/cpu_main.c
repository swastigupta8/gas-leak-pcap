/*
 * cpu_main.c - the CPU executable `gas2d`: --variant serial (one core) or omp (all cores).
 *
 *   ./build/gas2d --variant omp --nx 4000 --ny 4000 --steps 100 --threads 8
 */
#include "scenario.h"

#include <stdio.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {
    g2_config c;
    g2_params p;
    g2_weights k;
    if (g2_parse(argc, argv, &c, "omp") != 0 || g2_setup(&c, &p, &k) != 0) return 1;

    int use_omp = strcmp(c.variant, "omp") == 0;
    if (!use_omp && strcmp(c.variant, "serial") != 0) {
        fprintf(stderr, "--variant must be serial or omp\n");
        return 1;
    }
    int threads = 1;
#ifdef _OPENMP
    if (c.threads > 0) omp_set_num_threads(c.threads);
    threads = use_omp ? omp_get_max_threads() : 1;
#else
    if (use_omp) fprintf(stderr, "warning: built without OpenMP, running on one thread\n");
#endif

    float *a = g2_alloc(c.nx, c.ny), *b = g2_alloc(c.nx, c.ny);
    if (!a || !b) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    const int ld = c.nx + 2;
    g2_initial_block(&c, &p, a, ld, c.nx, c.ny, 0, 0);
    const size_t src = g2_idx(c.src_i + 1, c.src_j + 1, ld);
    const float s = g2_source_per_step(&c, &p);

    double t0 = g2_now();
    for (int n = 0; n < c.steps; ++n) {
        if (use_omp) g2_step_omp(a, b, c.nx, c.ny, k);
        else g2_step_serial(a, b, c.nx, c.ny, k);
        if (c.has_source) b[src] += s; /* the leak adds gas after the transport step */
        float *t = a; a = b; b = t;     /* ping-pong: the new field becomes the input */
    }
    double ms = c.steps > 0 ? (g2_now() - t0) * 1e3 / c.steps : 0.0;

    char extra[64];
    snprintf(extra, sizeof extra, "\"threads\":%d", threads);
    g2_report(&c, &p, ms, extra, a);
    g2_free(a);
    g2_free(b);
    return 0;
}
