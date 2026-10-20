"""
Correctness of the 2D core.

1. Every parallel version computes EXACTLY the serial answer (bit for bit): OpenMP, MPI on 1-4
   ranks with blocking and overlapped halo exchange, and the CUDA kernels run under the CPU
   emulator. Odd grid sizes make sure partial blocks and uneven MPI splits are exercised.
2. The physics is right: against the analytic solution of a drifting, spreading puff, pure
   diffusion is accurate to <0.2%, and with wind the error halves when the cell size halves
   (the first-order convergence the upwind scheme should show).
3. Mass is conserved, concentrations never go negative, unstable time steps are refused.
"""
import ctypes
import subprocess

import numpy as np
import pytest

from conftest import BUILD, mpiexec_cmd, run

# A leak off-centre, wind from the south-west: nothing in the problem is symmetric, so an
# indexing mistake (swapped x/y, wrong neighbour) cannot cancel out.
LEAK = ["--u", "2.5", "--v", "-1.2", "--D", "4"]
SIZES = [(97, 61), (64, 64), (33, 130), (1, 7)]


def leak_args(nx, ny, steps=40):
    return ["--nx", nx, "--ny", ny, "--steps", steps, "--source", f"{nx // 3},{2 * ny // 3},5000", *LEAK]


@pytest.mark.parametrize("nx,ny", SIZES)
def test_openmp_matches_serial(nx, ny, tmp_path):
    _, ref = run("gas2d", ["--variant", "serial", *leak_args(nx, ny)], tmp_path, name="s.bin")
    _, got = run("gas2d", ["--variant", "omp", "--threads", 2, *leak_args(nx, ny)], tmp_path, name="o.bin")
    assert np.array_equal(ref, got)


@pytest.mark.parametrize("ranks", [1, 2, 3, 4])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("nx,ny", [(97, 61), (40, 130)])
def test_mpi_matches_serial(ranks, overlap, nx, ny, tmp_path):
    _, ref = run("gas2d", ["--variant", "serial", *leak_args(nx, ny)], tmp_path, name="s.bin")
    args = leak_args(nx, ny) + (["--overlap"] if overlap else [])
    info, got = run("gas2d_mpi", args, tmp_path, prefix=mpiexec_cmd(ranks), name="m.bin")
    assert info["ranks"] == ranks
    assert np.array_equal(ref, got)


@pytest.mark.parametrize("kernel", ["naive", "shared", "coarse"])
@pytest.mark.parametrize("nx,ny", SIZES)
def test_gpu_kernels_emulated_match_serial(kernel, nx, ny, tmp_path):
    _, ref = run("gas2d", ["--variant", "serial", *leak_args(nx, ny, steps=15)], tmp_path, name="s.bin")
    _, got = run("gas2d_emu", ["--variant", kernel, *leak_args(nx, ny, steps=15)], tmp_path, name="e.bin")
    assert np.array_equal(ref, got)


def analytic(nx, ny, h, D, u, v, x0, y0, M, t):
    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h
    X, Y = np.meshgrid(x, y)
    return M / (4 * np.pi * D * t) * np.exp(-((X - x0 - u * t) ** 2 + (Y - y0 - v * t) ** 2) / (4 * D * t))


def puff_error(tmp_path, h, u, v, x0, y0):
    """Release a 10 s old puff, simulate 20 s more on a 400 m square map, return relative L2 error."""
    L, D, t0, T, M = 400.0, 10.0, 10.0, 20.0, 1e6
    n = int(L / h)
    base = ["--variant", "omp", "--nx", n, "--ny", n, "--h", h, "--D", D, "--u", u, "--v", v,
            "--gauss", f"{x0},{y0},{M},{t0}"]
    info, _ = run("gas2d", [*base, "--steps", 0], tmp_path)
    steps = int(round(T / info["dt"]))
    info, f = run("gas2d", [*base, "--steps", steps], tmp_path)
    exact = analytic(n, n, h, D, u, v, x0, y0, M, t0 + steps * info["dt"])
    return np.linalg.norm(f - exact) / np.linalg.norm(exact), info["mass"] / M, f


def test_pure_diffusion_matches_analytic(tmp_path):
    err, mass, f = puff_error(tmp_path, h=1.0, u=0.0, v=0.0, x0=200, y0=200)
    assert err < 2e-3
    assert abs(mass - 1) < 1e-4
    assert f.min() >= 0.0


def test_upwind_advection_converges_first_order(tmp_path):
    e2, _, _ = puff_error(tmp_path, h=2.0, u=3.0, v=1.0, x0=170, y0=190)
    e1, _, _ = puff_error(tmp_path, h=1.0, u=3.0, v=1.0, x0=170, y0=190)
    assert e1 < 0.06
    # First order: halving h should roughly halve the error (ratio ~0.5).
    assert 0.4 < e1 / e2 < 0.65


def test_leak_mass_equals_amount_released(tmp_path):
    steps, rate = 300, 5000.0
    info, f = run("gas2d", ["--variant", "omp", "--nx", 301, "--ny", 301, "--steps", steps,
                            "--source", f"150,150,{rate}"], tmp_path)
    released = rate * info["dt"] * steps * 1.0 ** 2  # mg/m^3/s x s x cell area (h = 1)
    assert abs(info["mass"] / released - 1) < 1e-4  # float rounding only; gas has not left the map
    assert f.min() >= 0.0


def test_unstable_time_step_is_refused(tmp_path):
    exe = BUILD / "gas2d"
    res = subprocess.run([str(exe), "--dt", "1.0", "--steps", "1"], capture_output=True, text=True)
    assert res.returncode != 0
    assert "unstable" in res.stderr


def test_dashboard_library_matches_executable(tmp_path):
    """The ctypes library the dashboard uses must run the same simulation as gas2d."""
    lib = ctypes.CDLL(str(BUILD / "libgas2d_cpu.so"))
    lib.g2s_create.restype = ctypes.c_void_p
    lib.g2s_create.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.g2s_set_params.restype = ctypes.c_float
    lib.g2s_set_params.argtypes = [ctypes.c_void_p] + [ctypes.c_float] * 5
    lib.g2s_set_source.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.g2s_step.restype = ctypes.c_double
    lib.g2s_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.g2s_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
    lib.g2s_destroy.argtypes = [ctypes.c_void_p]

    nx, ny, steps = 120, 80, 50
    sim = lib.g2s_create(nx, ny)
    dt = lib.g2s_set_params(sim, 1.0, 4.0, 2.5, -1.2, 0.0)
    lib.g2s_set_source(sim, nx // 3, 2 * ny // 3, ctypes.c_float(5000 * dt))
    lib.g2s_step(sim, steps)
    frame = np.zeros((ny, nx), np.float32)  # full resolution: one pixel per cell
    lib.g2s_frame(sim, frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), nx, ny)
    small = np.zeros((ny // 4, nx // 4), np.float32)  # 4 x 4 max-pooling
    lib.g2s_frame(sim, small.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), nx // 4, ny // 4)
    lib.g2s_destroy(sim)

    _, ref = run("gas2d", ["--variant", "serial", *leak_args(nx, ny, steps)], tmp_path)
    assert np.array_equal(frame, ref)
    assert np.array_equal(small, ref.reshape(ny // 4, 4, nx // 4, 4).max(axis=(1, 3)))
