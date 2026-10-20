/*
 * cuda_main.cu - `gas2d_cuda`: GPU benchmark and correctness run.
 *
 *   ./build/gas2d_cuda --variant coarse --nx 4000 --ny 4000 --steps 500
 *
 * Timing covers only the time steps (data is already on the GPU), measured with CUDA events.
 * A few warm-up steps run first so one-off costs (context creation, first-launch overhead) do
 * not distort the result; the field is then reset so the output matches the CPU run exactly.
 */
#include "cuda_backend.h"
#include "scenario.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    g2_config c;
    g2_params p;
    g2_weights k;
    if (g2_parse(argc, argv, &c, "coarse") != 0 || g2_setup(&c, &p, &k) != 0) return 1;
    int kernel = g2_kernel_from_name(c.variant);
    if (kernel < 0) {
        fprintf(stderr, "--variant must be naive, shared or coarse\n");
        return 1;
    }

    float *host = g2_alloc(c.nx, c.ny);
    if (!host) return 1;
    const int ld = c.nx + 2;
    g2_initial_block(&c, &p, host, ld, c.nx, c.ny, 0, 0);
    const long long src = c.has_source ? (long long)g2_idx(c.src_i + 1, c.src_j + 1, ld) : -1;
    const float s = g2_source_per_step(&c, &p);

    g2_gpu g;
    if (g2_gpu_create(&g, c.nx, c.ny) != 0) {
        fprintf(stderr, "not enough GPU memory for %d x %d\n", c.nx, c.ny);
        return 1;
    }
    g2_gpu_upload(&g, host);
    g2_gpu_run(&g, (g2_kernel)kernel, 5, k, src, s); /* warm-up */
    g2_gpu_zero(&g);
    g2_gpu_upload(&g, host);

    float ms = g2_gpu_run(&g, (g2_kernel)kernel, c.steps, k, src, s);
    g2_gpu_download(&g, host);
    double ms_step = c.steps > 0 ? ms / c.steps : 0.0;

    char name[128], extra[320];
    double peak;
    g2_gpu_info(name, sizeof name, &peak);
    double gbps = ms_step > 0 ? 8.0 * c.nx * (double)c.ny / (ms_step * 1e6) : 0.0;
    snprintf(extra, sizeof extra, "\"device\":\"%s\",\"peak_gbps\":%.4g,\"pct_of_peak\":%.4g", name, peak,
             peak > 0 ? 100.0 * gbps / peak : 0.0);
    g2_report(&c, &p, ms_step, extra, host);

    g2_gpu_destroy(&g);
    g2_free(host);
    return 0;
}
