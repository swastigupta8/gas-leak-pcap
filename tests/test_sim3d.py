"""
Smoke tests for the 3D simulator (sim3d/), run through its CPU build so they work without a GPU.
The GPU-vs-CPU comparison needs a GPU; it is in tests/test_gpu.py and skips without one.
"""
import json
import subprocess

import pytest

from conftest import ROOT, mpiexec_cmd

SIM3D = ROOT / "sim3d"


@pytest.fixture(scope="module", autouse=True)
def build_sim3d():
    subprocess.run(["make", "-C", str(SIM3D), "-j2"], check=True, capture_output=True)


def last_json(cmd, cwd):
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True, timeout=600).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_benchmark_runs_and_reports_throughput():
    r = last_json(["./bench_cpu", "48", "48", "12", "10", "2"], SIM3D / "build")
    assert r["backend"] == "cpu" and r["air_mlups"] > 0


@pytest.mark.parametrize("ranks", [1, 2])
def test_mpi_matches_single_process(ranks):
    # mpi_sim compares its gathered result with a single-process run and reports "pass".
    r = last_json(mpiexec_cmd(ranks) + ["./mpi_cpu", "48", "48", "12", "20"], SIM3D / "build")
    assert r["ranks"] == ranks and r["pass"] is True
