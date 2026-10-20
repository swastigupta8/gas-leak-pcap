"""
benchmark.py - runs every 2D variant and produces the tables and figures for the report.

    python scripts/benchmark.py                 # everything that is built
    python scripts/benchmark.py --quick         # small grids, for a smoke test
    python scripts/benchmark.py --skip-mpi --skip-gpu

Writes results/bench.json and results/*.png. Each measurement is one JSON line from the
executables in core2d/build (see core2d/include/scenario.h for what they report).

Experiments:
  1. variants  - serial, OpenMP, MPI, and the three GPU kernels on the largest grid
  2. openmp    - strong scaling: same grid, 1..N threads  -> speed-up and parallel efficiency
  3. mpi       - 1..P ranks, blocking vs overlapped halo exchange, compute vs communication time
  4. gpu       - each kernel across grid sizes, as achieved bandwidth vs the GPU's peak (roofline)
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "core2d" / "build"
OUT = ROOT / "results"


def run(cmd):
    cmd = [str(c) for c in cmd]  # Paths and numbers -> strings for the command line
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{res.stderr}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def mpiexec(n):
    exe = shutil.which("mpiexec") or shutil.which("mpirun")
    if exe is None:
        return None
    ver = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    extra = []
    # Open MPI 4.x reports itself as "OpenRTE", 5.x as "Open MPI"; MPICH needs no extra flags.
    if any(t in ver for t in ("Open MPI", "OpenRTE", "open-mpi.org")):
        extra.append("--oversubscribe")  # allow more ranks than CPU cores
        if os.geteuid() == 0:
            extra.append("--allow-run-as-root")
    return [exe, *extra, "-n", str(n)]


def grid(n, steps):
    return ["--nx", n, "--ny", n, "--steps", steps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-mpi", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true")
    a = ap.parse_args()

    N = 1000 if a.quick else 4000  # 4000 x 4000 = the synopsis target grid
    cpu_steps, gpu_steps = (20, 200) if a.quick else (20, 500)
    cores = os.cpu_count() or 1
    R = {"cores": cores, "grid": N}
    OUT.mkdir(exist_ok=True)

    print(f"1. variants on {N} x {N}")
    v = [run([BUILD / "gas2d", "--variant", "serial", *grid(N, cpu_steps)]),
         run([BUILD / "gas2d", "--variant", "omp", *grid(N, cpu_steps)])]
    gpu = (BUILD / "gas2d_cuda").exists() and not a.skip_gpu
    if gpu:
        for k in ("naive", "shared", "coarse"):
            v.append(run([BUILD / "gas2d_cuda", "--variant", k, *grid(N, gpu_steps)]))
    R["variants"] = v
    for r in v:
        print(f"   {r['variant']:8s} {r['ms_per_step']:9.3f} ms/step  {r['mlups']:9.0f} MLUPS  {r['realtime_factor']:8.1f}x real time")

    print("2. OpenMP strong scaling")
    threads = sorted({1, 2, 4, 8, 16, 32, cores} & set(range(1, cores + 1)))
    R["openmp"] = [run([BUILD / "gas2d", "--variant", "omp", "--threads", t, *grid(N, cpu_steps)]) for t in threads]

    if not a.skip_mpi and mpiexec(1):
        print("3. MPI")
        R["mpi"] = []
        for p in sorted({1, 2, 4, cores} & set(range(1, max(cores, 4) + 1))):
            for ov in (False, True):
                R["mpi"].append(run(mpiexec(p) + [BUILD / "gas2d_mpi", *grid(N, cpu_steps)] + (["--overlap"] if ov else [])))

    if gpu:
        print("4. GPU kernels across grid sizes")
        sizes = [256, 512, 1024, 2048, 4096] if not a.quick else [256, 512, 1024]
        R["gpu_sizes"] = [run([BUILD / "gas2d_cuda", "--variant", k, *grid(n, gpu_steps)])
                          for k in ("naive", "shared", "coarse") for n in sizes]

    (OUT / "bench.json").write_text(json.dumps(R, indent=1))
    plots(R)
    print(f"wrote {OUT / 'bench.json'} and figures in {OUT}")


def plots(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. throughput per variant (log scale: GPU and CPU differ by orders of magnitude)
    v = R["variants"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["variant"] for r in v], [r["mlups"] for r in v], color="#4878a8")
    ax.set_yscale("log")
    # Automatic log limits start just under the smallest bar, which makes it look empty; anchor
    # the axis at half the smallest value so bar heights compare honestly.
    ax.set_ylim(0.5 * min(r["mlups"] for r in v), 2.0 * max(r["mlups"] for r in v))  # room for labels
    ax.set_ylabel("million cell updates / s (log)")
    ax.set_title(f"{R['grid']} x {R['grid']} grid: throughput by implementation")
    for i, r in enumerate(v):
        ax.text(i, r["mlups"], f"{r['realtime_factor']:.1f}x real time", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "variants.png", dpi=150)

    # 2. OpenMP speed-up vs ideal
    o = R["openmp"]
    t1 = o[0]["ms_per_step"]
    fig, ax = plt.subplots(figsize=(5, 4))
    th = [r["threads"] for r in o]
    ax.plot(th, [t1 / r["ms_per_step"] for r in o], "o-", label="measured")
    ax.plot(th, th, "--", color="grey", label="ideal")
    ax.set_xlabel("threads")
    ax.set_ylabel("speed-up over 1 thread")
    ax.set_title("OpenMP strong scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "openmp_scaling.png", dpi=150)

    # 3. MPI compute vs exposed communication
    if R.get("mpi"):
        m = R["mpi"]
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [f"{r['ranks']}{' ov' if r['overlap'] else ''}" for r in m]
        ax.bar(labels, [r["compute_ms"] for r in m], label="compute")
        ax.bar(labels, [r["comm_ms"] for r in m], bottom=[r["compute_ms"] for r in m], label="halo exchange (exposed)")
        ax.set_xlabel("MPI ranks (ov = overlapped exchange)")
        ax.set_ylabel("ms per step (slowest rank)")
        ax.set_title("MPI: where the time goes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "mpi.png", dpi=150)

    # 4. GPU achieved bandwidth vs peak
    if R.get("gpu_sizes"):
        g = R["gpu_sizes"]
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in ("naive", "shared", "coarse"):
            rows = [r for r in g if r["variant"] == k]
            ax.plot([r["nx"] for r in rows], [r["gbps"] for r in rows], "o-", label=k)
        peak = g[0].get("peak_gbps", 0)
        if peak:
            ax.axhline(peak, ls="--", color="grey", label=f"peak {peak:.0f} GB/s")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("grid side (cells)")
        ax.set_ylabel("effective bandwidth (GB/s)")
        ax.set_title(f"GPU kernels on {g[0].get('device', 'GPU')}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "gpu_bandwidth.png", dpi=150)


if __name__ == "__main__":
    main()
