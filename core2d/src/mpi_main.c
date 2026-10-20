/*
 * mpi_main.c - distributed-memory version `gas2d_mpi`: the map is cut into a 2D grid of
 * rectangular blocks, one per MPI rank.
 *
 *     mpirun -np 4 ./build/gas2d_mpi --nx 4000 --ny 4000 --steps 100 [--overlap]
 *
 * Why a 2D (not strip) decomposition: each step a rank must receive the cells just outside its
 * block (the "halo"). With P ranks on an N x N grid, strips exchange 2N cells per rank no matter
 * how many ranks there are, while square blocks exchange about 4N/sqrt(P). Less communication per
 * unit of computation = better scaling.
 *
 * Halo exchange: rows are contiguous in memory and are sent directly. Columns are not (one value
 * per row, a full row apart), so an MPI derived datatype (MPI_Type_vector) describes the strided
 * column and MPI sends it without a manual copy. Ranks at the edge of the map have MPI_PROC_NULL
 * as their outer neighbour: those sends/receives do nothing, and the ghost cells stay 0.
 *
 * --overlap: post non-blocking sends/receives, update the cells that do not need any halo value
 * while messages are in flight, then wait and update the one-cell ring at the block edge. This
 * hides communication time behind computation.
 *
 * The final field is gathered on rank 0 (only when --out is given), and must be bit-identical to
 * the serial run: tests/test_correctness.py checks this for 1, 2, 3 and 4 ranks.
 */
#include "scenario.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Split n cells over p parts as evenly as possible; part c gets [lo, lo+len). */
static void split(int n, int p, int c, int *lo, int *len) {
    int base = n / p, rem = n % p;
    *len = base + (c < rem ? 1 : 0);
    *lo = c * base + (c < rem ? c : rem);
}

typedef struct {
    MPI_Comm comm;
    int north, south, east, west;
    int lx, ly, ld;
    MPI_Datatype col; /* one interior column: ly floats, ld apart */
} halo_ctx;

/* Blocking exchange of the four edges. Each Sendrecv pairs a send in one direction with the
 * matching receive from the opposite side, so no rank can deadlock waiting on another. */
static void exchange_blocking(float *a, const halo_ctx *h) {
    const int lx = h->lx, ly = h->ly, ld = h->ld;
    MPI_Sendrecv(a + g2_idx(1, ly, ld), lx, MPI_FLOAT, h->north, 0, a + g2_idx(1, 0, ld), lx, MPI_FLOAT,
                 h->south, 0, h->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a + g2_idx(1, 1, ld), lx, MPI_FLOAT, h->south, 1, a + g2_idx(1, ly + 1, ld), lx, MPI_FLOAT,
                 h->north, 1, h->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a + g2_idx(lx, 1, ld), 1, h->col, h->east, 2, a + g2_idx(0, 1, ld), 1, h->col, h->west, 2,
                 h->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a + g2_idx(1, 1, ld), 1, h->col, h->west, 3, a + g2_idx(lx + 1, 1, ld), 1, h->col, h->east, 3,
                 h->comm, MPI_STATUS_IGNORE);
}

/* Start a non-blocking exchange; the caller must MPI_Waitall(8, req, ...) before reading ghosts. */
static void exchange_start(float *a, const halo_ctx *h, MPI_Request req[8]) {
    const int lx = h->lx, ly = h->ly, ld = h->ld;
    MPI_Irecv(a + g2_idx(1, 0, ld), lx, MPI_FLOAT, h->south, 0, h->comm, &req[0]);
    MPI_Irecv(a + g2_idx(1, ly + 1, ld), lx, MPI_FLOAT, h->north, 1, h->comm, &req[1]);
    MPI_Irecv(a + g2_idx(0, 1, ld), 1, h->col, h->west, 2, h->comm, &req[2]);
    MPI_Irecv(a + g2_idx(lx + 1, 1, ld), 1, h->col, h->east, 3, h->comm, &req[3]);
    MPI_Isend(a + g2_idx(1, ly, ld), lx, MPI_FLOAT, h->north, 0, h->comm, &req[4]);
    MPI_Isend(a + g2_idx(1, 1, ld), lx, MPI_FLOAT, h->south, 1, h->comm, &req[5]);
    MPI_Isend(a + g2_idx(lx, 1, ld), 1, h->col, h->east, 2, h->comm, &req[6]);
    MPI_Isend(a + g2_idx(1, 1, ld), 1, h->col, h->west, 3, h->comm, &req[7]);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int size, wrank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);

    g2_config c;
    g2_params p;
    g2_weights k;
    if (g2_parse(argc, argv, &c, "mpi") != 0 || g2_setup(&c, &p, &k) != 0) MPI_Abort(MPI_COMM_WORLD, 1);

    /* Process grid: MPI_Dims_create returns the most square factorisation, larger factor first.
     * Give the larger factor to the longer side of the map. */
    int f[2] = {0, 0};
    MPI_Dims_create(size, 2, f);
    int pdims[2]; /* [0] = ranks along y (rows), [1] = ranks along x (columns) */
    pdims[0] = c.ny >= c.nx ? f[0] : f[1];
    pdims[1] = c.ny >= c.nx ? f[1] : f[0];
    int periods[2] = {0, 0}; /* the map does not wrap around */
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, pdims, periods, 1, &cart);
    int rank, coords[2];
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 2, coords);

    int x0, y0, lx, ly;
    split(c.nx, pdims[1], coords[1], &x0, &lx);
    split(c.ny, pdims[0], coords[0], &y0, &ly);
    if (lx < 1 || ly < 1) {
        if (rank == 0) fprintf(stderr, "too many ranks for a %d x %d grid\n", c.nx, c.ny);
        MPI_Abort(cart, 1);
    }

    halo_ctx h = {.comm = cart, .lx = lx, .ly = ly, .ld = lx + 2};
    MPI_Cart_shift(cart, 0, 1, &h.south, &h.north);
    MPI_Cart_shift(cart, 1, 1, &h.west, &h.east);
    MPI_Type_vector(ly, 1, h.ld, MPI_FLOAT, &h.col);
    MPI_Type_commit(&h.col);

    float *a = g2_alloc(lx, ly), *b = g2_alloc(lx, ly);
    if (!a || !b) MPI_Abort(cart, 1);
    g2_initial_block(&c, &p, a, h.ld, lx, ly, x0, y0);

    /* Only the rank whose block contains the leak adds gas. */
    int own_src = c.has_source && c.src_i >= x0 && c.src_i < x0 + lx && c.src_j >= y0 && c.src_j < y0 + ly;
    size_t src = own_src ? g2_idx(c.src_i - x0 + 1, c.src_j - y0 + 1, h.ld) : 0;
    const float s = g2_source_per_step(&c, &p);

    double t_comm = 0.0, t_comp = 0.0;
    MPI_Barrier(cart); /* start the clock together */
    double t0 = MPI_Wtime();
    for (int n = 0; n < c.steps; ++n) {
        if (!c.overlap) {
            double t1 = MPI_Wtime();
            exchange_blocking(a, &h);
            double t2 = MPI_Wtime();
            g2_step_rect(a, b, h.ld, 1, lx + 1, 1, ly + 1, k);
            t_comm += t2 - t1;
            t_comp += MPI_Wtime() - t2;
        } else {
            MPI_Request req[8];
            double t1 = MPI_Wtime();
            exchange_start(a, &h, req);
            /* Inner cells (one cell away from every block edge) need no halo values. */
            g2_step_rect(a, b, h.ld, 2, lx, 2, ly, k);
            double t2 = MPI_Wtime();
            MPI_Waitall(8, req, MPI_STATUSES_IGNORE);
            double t3 = MPI_Wtime();
            /* Now the edge ring: bottom row, top row, then the left and right columns between them. */
            g2_step_rect(a, b, h.ld, 1, lx + 1, 1, 2, k);
            if (ly > 1) g2_step_rect(a, b, h.ld, 1, lx + 1, ly, ly + 1, k);
            g2_step_rect(a, b, h.ld, 1, 2, 2, ly, k);
            if (lx > 1) g2_step_rect(a, b, h.ld, lx, lx + 1, 2, ly, k);
            t_comp += (t2 - t1) + (MPI_Wtime() - t3);
            t_comm += t3 - t2; /* only the waiting that could NOT be hidden */
        }
        if (own_src) b[src] += s;
        float *t = a; a = b; b = t;
    }
    double t_total = MPI_Wtime() - t0;

    /* The slowest rank sets the pace, so report maxima. */
    double loc[3] = {t_total, t_comp, t_comm}, mx[3];
    MPI_Reduce(loc, mx, 3, MPI_DOUBLE, MPI_MAX, 0, cart);
    double local_mass = g2_mass(a, lx, ly, c.h), mass;
    MPI_Reduce(&local_mass, &mass, 1, MPI_DOUBLE, MPI_SUM, 0, cart);

    /* Optional gather of the full field on rank 0 for verification. */
    float *global = NULL;
    if (c.out) {
        int meta[4] = {x0, y0, lx, ly};
        int *all = rank == 0 ? malloc(sizeof(int) * 4 * size) : NULL;
        MPI_Gather(meta, 4, MPI_INT, all, 4, MPI_INT, 0, cart);
        float *pack = malloc(sizeof(float) * (size_t)lx * ly);
        for (int j = 0; j < ly; ++j)
            for (int i = 0; i < lx; ++i) pack[(size_t)j * lx + i] = a[g2_idx(i + 1, j + 1, h.ld)];
        int *counts = NULL, *displs = NULL;
        float *recv = NULL;
        if (rank == 0) {
            counts = malloc(sizeof(int) * size);
            displs = malloc(sizeof(int) * size);
            int off = 0;
            for (int r = 0; r < size; ++r) {
                counts[r] = all[4 * r + 2] * all[4 * r + 3];
                displs[r] = off;
                off += counts[r];
            }
            recv = malloc(sizeof(float) * (size_t)off);
        }
        MPI_Gatherv(pack, lx * ly, MPI_FLOAT, recv, counts, displs, MPI_FLOAT, 0, cart);
        if (rank == 0) {
            global = g2_alloc(c.nx, c.ny);
            for (int r = 0; r < size; ++r) {
                int rx = all[4 * r], ry = all[4 * r + 1], rlx = all[4 * r + 2], rly = all[4 * r + 3];
                for (int j = 0; j < rly; ++j)
                    for (int i = 0; i < rlx; ++i)
                        global[g2_idx(rx + i + 1, ry + j + 1, c.nx + 2)] = recv[displs[r] + (size_t)j * rlx + i];
            }
            free(all);
            free(counts);
            free(displs);
            free(recv);
        }
        free(pack);
    }

    if (rank == 0) {
        const int st = c.steps > 0 ? c.steps : 1;
        char extra[256];
        int n = snprintf(extra, sizeof extra,
                         "\"ranks\":%d,\"dims\":[%d,%d],\"overlap\":%d,\"compute_ms\":%.6g,\"comm_ms\":%.6g", size,
                         pdims[0], pdims[1], c.overlap, mx[1] * 1e3 / st, mx[2] * 1e3 / st);
        /* When the field was gathered g2_report prints the mass itself; otherwise use the reduced sum. */
        if (!global) snprintf(extra + n, sizeof extra - (size_t)n, ",\"mass\":%.10g", mass);
        g2_report(&c, &p, mx[0] * 1e3 / st, extra, global);
    }

    g2_free(global);
    g2_free(a);
    g2_free(b);
    MPI_Type_free(&h.col);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
