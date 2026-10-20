"""
replay.py — runs the night of 7 May 2020 without the dashboard and saves results for the report.

    python server/replay.py --minutes 90            (uses site/vizag.npz, GPU if available)

Produces results/replay.npz (peak ground concentration map, arrival times) and results/replay.json
(table of localities), which the notebook turns into figures.
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
from simcore import Site, Simulator, GASES, pasquill_class

ROOT = Path(__file__).resolve().parent.parent
START_MIN, LEAK_MIN = 2 * 60 + 30, 2 * 60 + 42


def weather_at(site, minutes):
    hrs = site.meta.get("weather") or []
    if not hrs:
        return dict(wind=1.5, dir=225.0, temp=27.0, cloud=20, rain=0.0)
    t = (START_MIN + minutes) / 60.0
    k = int(min(max(math.floor(t), 0), len(hrs) - 2)); f = min(max(t - k, 0.0), 1.0)
    a, b = hrs[k], hrs[k + 1]
    s = (1 - f) * math.sin(math.radians(a["dir"])) + f * math.sin(math.radians(b["dir"]))
    c = (1 - f) * math.cos(math.radians(a["dir"])) + f * math.cos(math.radians(b["dir"]))
    return dict(wind=(1 - f) * a["wind"] + f * b["wind"], dir=math.degrees(math.atan2(s, c)) % 360,
                temp=(1 - f) * a["temp"] + f * b["temp"], cloud=(1 - f) * a["cloud"] + f * b["cloud"],
                rain=(1 - f) * (a.get("rain") or 0) + f * (b.get("rain") or 0))


def run(site, minutes=90, rate=10.0, height=12.0, backend="auto", spinup=1500, gas="styrene", log=print):
    sim = Simulator(site, backend=backend)
    g = GASES[gas]
    def set_weather(m):
        w = weather_at(site, m)
        hour = (START_MIN + m) / 60 % 24
        st = pasquill_class(w["wind"], w["cloud"], hour < 6 or hour >= 18.5, 30)
        sim.configure(wind=w["wind"], dir=w["dir"], temp=w["temp"], stability=st,
                      decay_per_hour=0.5 if w["rain"] > 0.5 else 0.0, gas=gas)
        return w, st
    w, st = set_weather(0)
    log(f"backend {sim.backend}, grid {sim.nx}x{sim.ny}x{sim.nz}, dt {sim.dt:.3f} s; weather at 02:30: "
        f"{w['wind']:.1f} m/s from {w['dir']:.0f} deg, class {st}")
    sim.gas_mode = -1; sim.step(spinup); sim.clear_gas()
    sim.gas_mode = 1 if sim.backend == "cuda" else 0
    peak = np.zeros((sim.ny, sim.nx), np.float32)
    arrive = np.full((sim.ny, sim.nx), np.nan, np.float32)        # minute when AEGL-1 first reached
    places = [dict(p, peak=0.0, first=None) for p in site.meta.get("places", [])
              if 0 <= p["x"] < site.nx and 0 <= p["y"] < site.ny]
    t_wall = time.time(); m = 0.0; chunk = 200; leak_on = False; series = []
    while m < minutes:
        if not leak_on and START_MIN + m >= LEAK_MIN:
            sim.set_sources([dict(x=site.meta["plant"]["x"], y=site.meta["plant"]["y"], height_m=height, rate_kgs=rate)])
            leak_on = True
        sim.step(chunk); m += chunk * sim.dt / 60
        if int(m) % 2 == 0: set_weather(m)
        gl = sim.ground_ppm()
        np.maximum(peak, gl, out=peak)
        new = np.isnan(arrive) & (gl >= g["aegl1"]); arrive[new] = m
        for p in places:
            v = float(gl[max(0, p["y"] - 1):p["y"] + 2, max(0, p["x"] - 1):p["x"] + 2].max())
            p["peak"] = max(p["peak"], v)
            if p["first"] is None and v >= g["aegl1"]: p["first"] = round(m, 1)
        clock = START_MIN + m
        series.append(dict(min=round(m, 2), clock=f"{int(clock // 60):02d}:{int(clock % 60):02d}", peak=float(gl.max()),
                           a1_km2=float((gl >= g["aegl1"]).sum()) * sim.dx ** 2 / 1e6,
                           a2_km2=float((gl >= g["aegl2"]).sum()) * sim.dx ** 2 / 1e6))
        if len(series) % 10 == 0:
            log(f"  {series[-1]['clock']} IST  ground peak {series[-1]['peak']:.0f} ppm, AEGL-1 area {series[-1]['a1_km2']:.2f} km²"
                f"  ({time.time() - t_wall:.0f} s wall)")
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / "replay.npz", peak=peak, arrive=arrive, terrain=site.terrain, buildings=site.buildings,
                        dx=sim.dx, plant=np.array([site.meta["plant"]["x"], site.meta["plant"]["y"]]))
    for p in places:
        p["level"] = 3 if p["peak"] >= g["aegl3"] else 2 if p["peak"] >= g["aegl2"] else 1 if p["peak"] >= g["aegl1"] else 0
        p["first_clock"] = None if p["first"] is None else \
            f"{int((START_MIN + p['first']) // 60):02d}:{int((START_MIN + p['first']) % 60):02d}"
    res = dict(minutes=minutes, rate_kgs=rate, wall_s=time.time() - t_wall, backend=sim.backend,
               places=[{k: p.get(k) for k in ("name", "dist_m", "reported", "peak", "first_clock", "level")} for p in places],
               series=series, aegl=[g["aegl1"], g["aegl2"], g["aegl3"]])
    (out / "replay.json").write_text(json.dumps(res, indent=1))
    log(f"done: {minutes} simulated minutes in {res['wall_s']:.0f} s")
    sim.lib.sim_free()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--site", default=str(ROOT / "site" / "vizag.npz"))
    a = ap.parse_args()
    site = Site.load(a.site) if Path(a.site).exists() else Site.synthetic()
    run(site, a.minutes, a.rate, backend=a.backend)
