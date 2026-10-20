"""
experiments.py - the 3D simulator's experiments and report figures (formerly notebook cells).

    python sim3d/experiments.py site      download the real Vizag data (needs internet, 1-3 min)
    python sim3d/experiments.py check     GPU result vs CPU result on the same test case
    python sim3d/experiments.py bench     CPU 1 core vs CPU all cores vs GPU naive vs GPU shared
    python sim3d/experiments.py mpi       map split across 1, 2, 4 processes; must match one process
    python sim3d/experiments.py replay    replay the night of 7 May 2020 and map the worst exposure
    python sim3d/experiments.py all       everything above except `site`

Uses the GPU builds (make -C sim3d cuda) when an NVIDIA GPU is available, otherwise the CPU
builds, so every command also works on a laptop without a GPU. Without site/vizag.npz the
synthetic test valley is used. Results and figures go to sim3d/results/.

Plotting code is Dhruvishi's, from the original notebook. Changes: explicit file paths, a CPU
fallback when there is no GPU, and a fixed log-axis limit that had hidden part of a bar.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD, RESULTS, SITE = HERE / "build", HERE / "results", HERE / "site" / "vizag.npz"
sys.path.insert(0, str(HERE / "server"))


def gpu_available():
    """True only if the GPU programs are built AND a working NVIDIA driver is present."""
    smi = shutil.which("nvidia-smi")
    ok = smi is not None and subprocess.run([smi], capture_output=True).returncode == 0
    return ok and (BUILD / "bench_cuda").exists()


def run_json(args, cwd=BUILD):
    res = subprocess.run([str(a) for a in args], capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        raise SystemExit(f"{' '.join(map(str, args))} failed:\n{res.stderr}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def mpirun(n):
    exe = shutil.which("mpirun") or shutil.which("mpiexec")
    if exe is None:
        raise SystemExit("MPI is not installed (sudo apt install openmpi-bin libopenmpi-dev)")
    ver = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout
    extra = []
    # Open MPI 4.x reports itself as "OpenRTE", 5.x as "Open MPI"; MPICH needs no extra flags.
    if any(t in ver for t in ("Open MPI", "OpenRTE", "open-mpi.org")):
        extra.append("--oversubscribe")  # allow more ranks than cores (laptops)
        if os.geteuid() == 0:
            extra.append("--allow-run-as-root")
    return [exe, *extra, "-np", str(n)]


def site_grid():
    """Grid size the simulator would use for the current site (real or synthetic)."""
    from simcore import Site
    if SITE.exists():
        d = np.load(SITE)
        terrain, buildings, dx = d["terrain"], d["buildings"], float(d["dx"])
    else:
        s = Site.synthetic()
        terrain, buildings, dx = s.terrain, s.buildings, s.dx
    top = float((terrain + buildings).max())
    nz = int(min(64, max(8, math.ceil((top + 80) / dx))))
    return terrain.shape[1], terrain.shape[0], nz


def plt():
    import matplotlib
    matplotlib.use("Agg")  # write files; no window needed
    import matplotlib.pyplot as p
    return p


def cmd_site(a):
    subprocess.run([sys.executable, str(HERE / "data" / "prepare_site.py"), "--size", "5000", "--dx", "10"], check=True)


def cmd_check(a):
    if not gpu_available():
        print("check: skipped (no GPU build or no NVIDIA driver; this compares GPU against CPU)")
        return True
    run_json(["./bench_cuda", "check"])
    run_json(["./bench_cpu", "check"])
    g = np.fromfile(BUILD / "check_cuda.bin", np.float32)
    c = np.fromfile(BUILD / "check_cpu.bin", np.float32)
    n = len(g) // 2
    dC, dU = np.abs(g[:n] - c[:n]).max(), np.abs(g[n:] - c[n:]).max()
    ok = bool(dC < 1e-2 * c[:n].max() and dU < 1e-3)
    print(f"check: gas max difference {dC:.3g} ppm (peak {c[:n].max():.0f} ppm), wind {dU:.3g} lattice units"
          f" -> {'PASS' if ok else 'FAIL: GPU and CPU disagree'}")
    return ok


def cmd_bench(a):
    p = plt()
    gpu = gpu_available()
    NX, NY, NZ = site_grid()
    print(f"bench: site grid {NX} x {NY} x {NZ} = {NX * NY * NZ / 1e6:.1f} million boxes, GPU: {gpu}")
    cpu1 = run_json(["./bench_cpu", NX, NY, NZ, 2, 1])
    cpuN = run_json(["./bench_cpu", NX, NY, NZ, 2])
    rows = [("CPU 1 core", cpu1["air_ms"], cpu1["gas_naive_ms"]),
            (f"CPU {cpuN['threads']} cores", cpuN["air_ms"], cpuN["gas_naive_ms"])]
    res = dict(site_grid=[NX, NY, NZ], cpu_serial=cpu1, cpu_openmp=cpuN,
               cpu_air_ms=cpu1["air_ms"], cpu_gas_ms=cpu1["gas_naive_ms"])
    if gpu:
        g = run_json(["./bench_cuda", NX, NY, NZ, 100])
        res["gpu"] = g
        rows += [("GPU naive", g["air_ms"], g["gas_naive_ms"]), ("GPU shared", g["air_ms"], g["gas_shared_ms"])]
    for r in rows:
        print(f"   {r[0]:14s} air {r[1]:9.3f} ms  gas {r[2]:8.3f} ms per step")

    sizes = [64, 128, 256, 384] if gpu else [64, 128]
    scal = []
    for n in sizes:
        c = run_json(["./bench_cpu", n, n, 24, 3])
        s = dict(n=n, cpu=c)
        if gpu:
            s["gpu"] = run_json(["./bench_cuda", n, n, 24, 100])
        scal.append(s)
    res["scaling"] = scal
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "bench.json").write_text(json.dumps(res, indent=1))  # the dashboard shows these numbers

    fig, ax = p.subplots(1, 3, figsize=(17, 4.5))
    lab, air, gas = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    ax[0].bar(lab, air, label="air (LBM)")
    ax[0].bar(lab, gas, bottom=air, label="gas")
    ax[0].set_yscale("log")
    # Matplotlib's automatic log limits can start above the smallest bar and hide it; anchor the
    # axis well below the smallest segment so every bar is visible.
    ax[0].set_ylim(bottom=0.5 * min(min(air), min(g for g in gas if g > 0)))
    ax[0].set_ylabel("ms per time step (log scale)")
    ax[0].set_title("Time per step on the site grid")
    ax[0].legend()
    sp = [(air[0] + gas[0]) / (x + y) for x, y in zip(air, gas)]
    ax[1].bar(lab, sp, color=["#888", "#aaa", "#e98", "#e50"][: len(lab)])
    ax[1].set_title("Speed-up vs CPU 1 core")
    for i, v in enumerate(sp):
        ax[1].text(i, v, f"{v:.0f}x" if v >= 10 else f"{v:.1f}x", ha="center", va="bottom")
    ax[2].plot(sizes, [s["cpu"]["air_mlups"] for s in scal], "o-", label="CPU (all cores)")
    if gpu:
        ax[2].plot(sizes, [s["gpu"]["air_mlups"] for s in scal], "o-", label="GPU")
    ax[2].set_xlabel("grid width (boxes), 24 tall")
    ax[2].set_ylabel("million boxes updated per second")
    ax[2].set_yscale("log")
    ax[2].legend()
    ax[2].set_title("Throughput vs grid size")
    p.tight_layout()
    p.savefig(RESULTS / "benchmarks.png", dpi=140)
    if gpu and res["gpu"]["gas_shared_ms"] > 0:
        print(f"bench: gas kernel, shared memory is {res['gpu']['gas_naive_ms'] / res['gpu']['gas_shared_ms']:.2f}x "
              "the speed of naive on this GPU")
    print(f"bench: wrote {RESULTS / 'bench.json'} and benchmarks.png")


def cmd_mpi(a):
    p = plt()
    exe = "./mpi_cuda" if gpu_available() and (BUILD / "mpi_cuda").exists() else "./mpi_cpu"
    rows = []
    for n in [1, 2, 4]:
        r = run_json(mpirun(n) + [exe, 256, 256, 24, 100])
        rows.append(r)
        print(f"   {n} ranks ({r['backend']}): compute {r['compute_ms']:.2f} ms, halo {r['comm_ms']:.2f} ms, "
              f"matches single run: {r['pass']}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "mpi.json").write_text(json.dumps(rows, indent=1))
    p.figure(figsize=(6, 4))
    x = [str(r["ranks"]) for r in rows]
    p.bar(x, [r["compute_ms"] for r in rows], label="compute")
    p.bar(x, [r["comm_ms"] for r in rows], bottom=[r["compute_ms"] for r in rows], label="halo exchange")
    p.xlabel("MPI processes")
    p.ylabel("ms per step")
    p.legend()
    p.title("MPI: compute vs communication")
    p.savefig(RESULTS / "mpi.png", dpi=140)
    return all(r["pass"] for r in rows)


def cmd_replay(a):
    p = plt()
    from matplotlib.colors import LogNorm
    from simcore import Site
    subprocess.run([sys.executable, "replay.py", "--minutes", str(a.minutes)], cwd=HERE / "server", check=True)
    r = json.loads((RESULTS / "replay.json").read_text())
    z = np.load(RESULTS / "replay.npz")
    meta = json.loads(str(np.load(SITE)["meta"])) if SITE.exists() else Site.synthetic().meta
    fig, ax = p.subplots(1, 2, figsize=(15, 6.5))
    pk = np.maximum(z["peak"], 0.1)
    im = ax[0].imshow(pk, origin="lower", norm=LogNorm(0.1, 1e4), cmap="inferno")
    if z["peak"].max() > r["aegl"][0]:  # contours need the field to reach at least one level
        ax[0].contour(z["peak"], levels=r["aegl"], colors=["#f5d547", "#ff8a2a", "#ff3b3b"], linewidths=1.2)
    ax[0].contour(z["terrain"], levels=10, colors="w", linewidths=0.3, alpha=0.4)
    p.colorbar(im, ax=ax[0], label="peak ppm at breathing height")
    ax[0].set_title(f"Worst concentration, first {a.minutes:g} min (AEGL-1/2/3 contours)")
    im2 = ax[1].imshow(z["arrive"], origin="lower", cmap="viridis")
    p.colorbar(im2, ax=ax[1], label="minutes after 02:30 when AEGL-1 (20 ppm) was first reached")
    ax[1].set_title("When the gas arrived")
    for axis in ax:
        axis.plot(*z["plant"], "c^", ms=10)
        for pl in meta.get("places", []):
            rep = pl.get("reported", False)  # synthetic places carry no "reported" flag
            axis.plot(pl["x"], pl["y"], "o", mfc="none", mec="w" if rep else "#888", ms=6)
            if rep:
                axis.text(pl["x"] + 3, pl["y"] + 3, pl["name"], color="w", fontsize=7)
    p.tight_layout()
    p.savefig(RESULTS / "replay.png", dpi=140)
    print(f"{'Place':28s} {'km':>5s} {'peak ppm':>9s} {'AEGL-1 at':>9s}  reported")
    for pl in sorted(r["places"], key=lambda q: q["dist_m"] or 0):
        print(f"{pl['name'][:28]:28s} {(pl['dist_m'] or 0) / 1000:5.1f} {pl['peak']:9.0f} "
              f"{pl['first_clock'] or '-':>9s}  {'*' if pl['reported'] else ''}")
    print(f"replay: wrote {RESULTS / 'replay.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["site", "check", "bench", "mpi", "replay", "all"])
    ap.add_argument("--minutes", type=float, default=60, help="replay length in simulated minutes")
    a = ap.parse_args()
    if not (BUILD / "bench_cpu").exists():
        raise SystemExit("build first: make -C sim3d (and make -C sim3d cuda on a GPU machine)")
    ok = True
    if a.command == "site":
        cmd_site(a)
    if a.command in ("check", "all"):
        ok &= cmd_check(a)
    if a.command in ("bench", "all"):
        cmd_bench(a)
    if a.command in ("mpi", "all"):
        ok &= cmd_mpi(a)
    if a.command in ("replay", "all"):
        cmd_replay(a)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
