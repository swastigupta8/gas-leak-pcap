/*
 * gas2d_lib.h - the C interface the dashboard (Python, via ctypes) uses to drive a live
 * simulation. Two builds implement it with identical behaviour:
 *     libgas2d_cpu.so   OpenMP on the CPU          (any laptop)
 *     libgas2d_cuda.so  the CUDA kernels on a GPU  (Colab T4, or a local NVIDIA card)
 * The dashboard picks whichever loads, so the same front end demonstrates the speed-up.
 */
#ifndef GAS2D_LIB_H
#define GAS2D_LIB_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct g2_sim g2_sim;

g2_sim *g2s_create(int nx, int ny);
void g2s_destroy(g2_sim *s);
/* Physical parameters. dt <= 0 means 0.9 x the stability limit. Returns the dt used, or -1 if the
 * requested dt is unstable (parameters unchanged). */
float g2s_set_params(g2_sim *s, float h, float D, float u, float v, float dt);
/* Leak at interior cell (i, j) adding `per_step` mg/m^3 to it every step; i < 0 switches it off. */
void g2s_set_source(g2_sim *s, int i, int j, float per_step);
void g2s_clear(g2_sim *s);                        /* back to clean air */
double g2s_step(g2_sim *s, int n);                /* run n steps; returns elapsed ms */
void g2s_frame(g2_sim *s, float *out, int fx, int fy); /* max-pooled fx x fy image */
const char *g2s_backend(void);                    /* "cpu-openmp" or "cuda:<device>" */

#ifdef __cplusplus
}
#endif

#endif
