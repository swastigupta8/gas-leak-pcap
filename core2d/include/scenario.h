/*
 * scenario.h - command-line options, initial conditions and result reporting, shared by every
 * executable (serial/OpenMP, MPI, CUDA, CUDA emulator) so they all run exactly the same experiment.
 *
 * Common options:
 *   --nx N --ny N        grid size in cells                 (default 1024 x 1024)
 *   --steps N            time steps to run                  (default 200)
 *   --h M                cell size in metres                (default 1)
 *   --D M2S              eddy diffusivity, m^2/s            (default 10)
 *   --u MS --v MS        wind, m/s (east, north)            (default 3, 1)
 *   --dt S               time step; default 0.9 x stability limit
 *   --source I,J,RATE    continuous leak at interior cell (I,J), RATE in mg/m^3 per second
 *                        added to that cell                 (default: centre, 1e4)
 *   --gauss X,Y,M,T0     instead of a leak: start from the analytic puff of M mg/m released at
 *                        (X,Y) metres, T0 seconds old (used by the validation test)
 *   --out FILE           write the final interior field as raw float32, ny rows of nx values
 *   --variant NAME       which algorithm (meaning depends on the executable)
 *   --threads N          OpenMP threads (CPU executables)
 *   --overlap            MPI: overlap halo exchange with interior computation
 *
 * Each executable prints exactly one JSON line on stdout with its timings.
 */
#ifndef SCENARIO_H
#define SCENARIO_H

#include "gas2d.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char variant[32];
    int nx, ny, steps, threads, overlap;
    float h, D, u, v, dt;
    int has_source, src_i, src_j; /* 0-based interior coordinates */
    float src_rate;               /* mg/m^3 per second            */
    int has_gauss;
    double gx, gy, gM, gt0;
    const char *out;
} g2_config;

/* Parse argv into c (defaults first). Prints a message and returns -1 on bad input. */
int g2_parse(int argc, char **argv, g2_config *c, const char *default_variant);

/* Physical parameters from the config, choosing dt if not given. Returns -1 if unstable. */
int g2_setup(const g2_config *c, g2_params *p, g2_weights *k);

/* Fill the initial field of a local block (first interior cell = global (gx0, gy0)). */
void g2_initial_block(const g2_config *c, const g2_params *p, float *C, int ld, int lx, int ly, int gx0,
                      int gy0);

/* Amount added to the source cell each step (mg/m^3), or 0 if there is no source. */
float g2_source_per_step(const g2_config *c, const g2_params *p);

/*
 * Print the one-line JSON result. ms_per_step is the measured time per step; `extra` is
 * additional JSON fields (without braces, may be NULL or ""). If `global` is non-NULL it is the
 * full padded field: its mass is reported and it is written to c->out if requested.
 */
void g2_report(const g2_config *c, const g2_params *p, double ms_per_step, const char *extra,
               const float *global);

/* Write the interior of a padded nx x ny field as raw float32. Returns 0 on success. */
int g2_write_field(const char *path, const float *C, int nx, int ny);

#ifdef __cplusplus
}
#endif

#endif /* SCENARIO_H */
