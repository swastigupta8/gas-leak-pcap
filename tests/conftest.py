"""Shared fixtures: build the 2D core once, and helpers to run its executables."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core2d"
BUILD = CORE / "build"


@pytest.fixture(scope="session", autouse=True)
def build():
    """Compile every CPU target before any test runs (fails loudly if the build breaks)."""
    subprocess.run(["make", "-C", str(CORE), "-j2"], check=True, capture_output=True)


def mpiexec_cmd(n):
    """mpiexec prefix for n ranks. Open MPI refuses more ranks than cores unless told otherwise."""
    exe = shutil.which("mpiexec") or shutil.which("mpirun")
    if exe is None:
        pytest.skip("MPI is not installed")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    # Open MPI 4.x reports itself as "OpenRTE", 5.x as "Open MPI"; MPICH needs no extra flags.
    open_mpi = any(t in version for t in ("Open MPI", "OpenRTE", "open-mpi.org"))
    extra = ["--oversubscribe"] if open_mpi else []  # allow more ranks than CPU cores
    if open_mpi and os.geteuid() == 0:
        extra.append("--allow-run-as-root")  # containers often run as root
    return [exe, *extra, "-n", str(n)]


def run(exe, args, tmp_path, prefix=None, name="out.bin"):
    """Run a core2d executable; return (parsed JSON line, final field as an (ny, nx) array)."""
    out = tmp_path / name
    cmd = (prefix or []) + [str(BUILD / exe), *map(str, args), "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(res.stdout.strip().splitlines()[-1])
    field = np.fromfile(out, np.float32).reshape(info["ny"], info["nx"])
    return info, field
