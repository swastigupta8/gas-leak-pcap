"""
GPU tests: the real CUDA builds against the CPU. Skipped automatically when there is no NVIDIA
GPU or the CUDA targets are not built (`make cuda`), so the suite still passes on any laptop.

The GPU may fuse multiply-adds (FMA), which rounds differently from the CPU, so results are
compared with a tolerance relative to the peak value instead of bit for bit.
"""
import ctypes
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import BUILD, ROOT, run


def have_gpu():
    smi = shutil.which("nvidia-smi")
    return smi is not None and subprocess.run([smi], capture_output=True).returncode == 0


HAVE_GPU = have_gpu()


def needs(path):
    """Skip with the reason that actually applies: no GPU at all, or GPU present but not built."""
    if not HAVE_GPU:
        return pytest.mark.skip(reason="no NVIDIA GPU / driver on this machine")
    return pytest.mark.skipif(not path.exists(), reason=f"{path.name} not built: run `make cuda`")


ARGS = ["--nx", 1000, "--ny", 700, "--steps", 400, "--source", "300,450,5000", "--u", 2.5, "--v", -1.2, "--D", 4]


@needs(BUILD / "gas2d_cuda")
@pytest.mark.parametrize("kernel", ["naive", "shared", "coarse"])
def test_cuda_kernel_matches_serial(kernel, tmp_path):
    _, ref = run("gas2d", ["--variant", "serial", *ARGS], tmp_path, name="s.bin")
    info, got = run("gas2d_cuda", ["--variant", kernel, *ARGS], tmp_path, name="g.bin")
    assert np.abs(got - ref).max() <= 1e-4 * ref.max()
    assert abs(info["mass"] / (ref.sum() * 1.0) - 1) < 1e-4  # h = 1 m, so mass = sum of cells
    assert info["ms_per_step"] > 0


@needs(BUILD / "libgas2d_cuda.so")
def test_dashboard_gpu_library_matches_cpu_library():
    def sim(name):
        lib = ctypes.CDLL(str(BUILD / name))
        lib.g2s_create.restype = ctypes.c_void_p
        lib.g2s_create.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.g2s_set_params.restype = ctypes.c_float
        lib.g2s_set_params.argtypes = [ctypes.c_void_p] + [ctypes.c_float] * 5
        lib.g2s_set_source.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.g2s_step.restype = ctypes.c_double
        lib.g2s_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.g2s_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        lib.g2s_destroy.argtypes = [ctypes.c_void_p]
        s = lib.g2s_create(640, 480)
        dt = lib.g2s_set_params(s, 1.0, 10.0, 3.0, 1.0, 0.0)
        lib.g2s_set_source(s, 200, 240, ctypes.c_float(1e4 * dt))
        lib.g2s_step(s, 300)
        out = np.zeros((120, 160), np.float32)
        lib.g2s_frame(s, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 160, 120)
        lib.g2s_destroy(s)
        return out

    cpu, gpu = sim("libgas2d_cpu.so"), sim("libgas2d_cuda.so")
    assert np.abs(gpu - cpu).max() <= 1e-4 * cpu.max()


@needs(ROOT / "sim3d" / "build" / "bench_cuda")
def test_sim3d_gpu_matches_cpu():
    res = subprocess.run([sys.executable, str(ROOT / "sim3d" / "experiments.py"), "check"],
                         capture_output=True, text=True, timeout=1200)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PASS" in res.stdout
