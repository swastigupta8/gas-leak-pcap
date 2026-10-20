/*
 * emu_main.c - `gas2d_emu`: the GPU kernels executed on the CPU (see include/cuda_emu.h).
 * Slow on purpose; it exists to prove the kernels compute the same thing as the serial code.
 *
 *   ./build/gas2d_emu --variant shared --nx 333 --ny 97 --steps 20 --out emu.bin
 */
#include "cuda_emu.h"
#include "scenario.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    g2_config c;
    g2_params p;
    g2_weights k;
    if (g2_parse(argc, argv, &c, "shared") != 0 || g2_setup(&c, &p, &k) != 0) return 1;

    void (*kernel)(const float *, float *, int, int, g2_weights, long long, float) = NULL;
    if (strcmp(c.variant, "naive") == 0) kernel = emu_naive;
    else if (strcmp(c.variant, "shared") == 0) kernel = emu_shared;
    else if (strcmp(c.variant, "coarse") == 0) kernel = emu_coarse;
    else {
        fprintf(stderr, "--variant must be naive, shared or coarse\n");
        return 1;
    }

    float *a = g2_alloc(c.nx, c.ny), *b = g2_alloc(c.nx, c.ny);
    if (!a || !b) return 1;
    const int ld = c.nx + 2;
    g2_initial_block(&c, &p, a, ld, c.nx, c.ny, 0, 0);
    const long long src = c.has_source ? (long long)g2_idx(c.src_i + 1, c.src_j + 1, ld) : -1;
    const float s = g2_source_per_step(&c, &p);

    double t0 = g2_now();
    for (int n = 0; n < c.steps; ++n) {
        kernel(a, b, c.nx, c.ny, k, src, s); /* the leak is added inside the kernel, as on the GPU */
        float *t = a; a = b; b = t;
    }
    double ms = c.steps > 0 ? (g2_now() - t0) * 1e3 / c.steps : 0.0;

    /* Report as emu_<name>; variant names are short (naive/shared/coarse), so this always fits. */
    char variant[sizeof c.variant];
    snprintf(variant, sizeof variant, "emu_%.20s", c.variant);
    memcpy(c.variant, variant, sizeof variant);
    g2_report(&c, &p, ms, "\"emulated\":true", a);
    g2_free(a);
    g2_free(b);
    return 0;
}
