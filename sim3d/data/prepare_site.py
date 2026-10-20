"""
prepare_site.py — downloads the REAL data for the simulation area and saves it as site/vizag.npz

Run it once (needs internet; Colab has it):     python data/prepare_site.py

What it fetches (all free, no API keys):
  1. Terrain height   — Copernicus 90 m elevation model, via the Open-Meteo elevation API
  2. Buildings        — OpenStreetMap footprints + heights/levels, via the Overpass API
  3. Localities       — OpenStreetMap place names (villages, colonies) inside the area
  4. Weather          — hourly ERA5 reanalysis for the night of 6–7 May 2020, via Open-Meteo archive

If a download fails, the script says so and carries on with a safe fallback (flat ground,
no buildings, assumed weather), and records that in the file so the dashboard can show it.
"""
import json, math, sys, time, argparse
from pathlib import Path
import urllib.request, urllib.parse
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "PCAP-student-project-vizag-sim/1.0 (educational)"}

# LG Polymers plant, R.R. Venkatapuram, Visakhapatnam (coordinates from the Wikipedia article on the
# 2020 Visakhapatnam gas leak). The exact position of tank M6 inside the 213-acre site is not public,
# so the leak is placed at these coordinates.
PLANT = dict(lat=17.75528, lon=83.20889, name="LG Polymers (tank M6)")

# Localities named in the High Power Committee report (Section 1.3) and press coverage as affected.
REPORTED_AFFECTED = ["R. R. Venkatapuram", "Venkatapuram", "Venkatadri Gardens", "Kamparapalem",
                     "Padmanabha Nagar", "Nandamuri Nagar", "Srinagar", "B.C. Colony", "Padmapuram",
                     "Gopalapatnam", "Ajanta Park"]

# Known facts used for comparison on the dashboard (with sources)
FACTS = dict(
    release_start_ist="02:42 (vapour first seen on CCTV), alarm 02:54",
    deaths=12, hospitalised=585, affected_radius_km=3.0,
    est_release_tonnes=800, est_rate_kgs=10, est_duration_h=20,
    observed_cloud_height_m="about 2.4 to 4",
    sources=[
        ["High Power Committee report (2020)", "https://safteng.net/wp-content/uploads/2020/07/HPCR_on_Styrene_vapour_release_accident_at_LG_Polymers.pdf"],
        ["AIChE case study: runaway styrene at LG Polymers", "https://www.aiche.org/sites/default/files/docs/book-pages/3_chef_case_study_-_runaway_styrene_at_lg_polymer_0.pdf"],
        ["Wikipedia: Visakhapatnam gas leak", "https://en.wikipedia.org/wiki/Visakhapatnam_gas_leak"],
        ["US EPA AEGL values for styrene", "https://www.epa.gov/aegl/styrene-results-aegl-program"],
        ["Buoyancy-modified FLEXPART-WRF study of the leak", "https://link.springer.com/article/10.1007/s10666-026-10166-x"],
    ])


def get_json(url, data=None, timeout=90, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"    attempt {k+1} failed: {e}")
            time.sleep(2 + 3 * k)
    return None


class Grid:
    """Local flat-earth grid centred on the plant: x = metres east, y = metres north."""
    def __init__(self, lat0, lon0, size_m, dx):
        self.lat0, self.lon0, self.dx = lat0, lon0, dx
        self.n = int(round(size_m / dx))
        self.half = self.n * dx / 2
        self.mx = 111320.0 * math.cos(math.radians(lat0))   # metres per degree longitude
        self.my = 110574.0                                    # metres per degree latitude

    def to_xy(self, lat, lon):      # -> metres from the south-west corner
        return (lon - self.lon0) * self.mx + self.half, (lat - self.lat0) * self.my + self.half

    def to_latlon(self, x, y):
        return self.lat0 + (y - self.half) / self.my, self.lon0 + (x - self.half) / self.mx

    def bbox(self):
        s, w = self.to_latlon(0, 0); n, e = self.to_latlon(self.n * self.dx, self.n * self.dx)
        return s, w, n, e


# ------------------------------------------------------------------------------------------
def fetch_terrain(g, samples=48):
    print(f"[1/4] terrain: sampling {samples}x{samples} elevation points ...")
    xs = (np.arange(samples) + 0.5) / samples * g.n * g.dx
    pts = [(x, y) for y in xs for x in xs]
    lat_lon = [g.to_latlon(x, y) for x, y in pts]
    elev = []
    for k in range(0, len(lat_lon), 100):
        chunk = lat_lon[k:k + 100]
        url = ("https://api.open-meteo.com/v1/elevation?latitude=" + ",".join(f"{a:.5f}" for a, _ in chunk)
               + "&longitude=" + ",".join(f"{b:.5f}" for _, b in chunk))
        j = get_json(url)
        if not j or "elevation" not in j:
            print("    elevation service unavailable -> using flat ground")
            return np.zeros((g.n, g.n), np.float32), "flat (elevation download failed)"
        elev += j["elevation"]
        time.sleep(0.3)
    E = np.array(elev, np.float32).reshape(samples, samples)
    # bilinear interpolation from the coarse samples to the simulation grid
    c = (np.arange(g.n) + 0.5) / g.n * samples - 0.5
    c = np.clip(c, 0, samples - 1.001)
    i0 = np.floor(c).astype(int); t = c - i0
    rows = E[i0] * (1 - t)[:, None] + E[i0 + 1] * t[:, None]
    T = rows[:, i0] * (1 - t)[None, :] + rows[:, i0 + 1] * t[None, :]
    print(f"    elevation {E.min():.0f} to {E.max():.0f} m above sea level")
    return T.astype(np.float32), "Copernicus GLO-90 DEM via Open-Meteo"


def fetch_buildings(g, max_polys=8000):
    print("[2/4] buildings: asking OpenStreetMap (Overpass) ...")
    s, w, n, e = g.bbox()
    q = f'[out:json][timeout:120];way["building"]({s},{w},{n},{e});out geom tags;'
    j = None
    for url in ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]:
        j = get_json(url, data=urllib.parse.urlencode({"data": q}).encode(), timeout=150, tries=2)
        if j: break
    H = np.zeros((g.n, g.n), np.float32)
    polys = []
    if not j:
        print("    Overpass unavailable -> no buildings")
        return H, polys, "none (download failed)"
    from matplotlib.path import Path as MPath
    yy, xx = np.mgrid[0:g.n, 0:g.n]
    centres = np.stack([(xx.ravel() + 0.5) * g.dx, (yy.ravel() + 0.5) * g.dx], 1)
    for el in j.get("elements", []):
        geom = el.get("geometry")
        if not geom or len(geom) < 3: continue
        tags = el.get("tags", {})
        h = None
        try:
            if "height" in tags: h = float(str(tags["height"]).split()[0])
            elif "building:levels" in tags: h = 3.2 * float(tags["building:levels"])
        except ValueError:
            h = None
        if h is None: h = 7.0 if tags.get("building") not in ("industrial", "warehouse") else 10.0
        pts = np.array([g.to_xy(p["lat"], p["lon"]) for p in geom], np.float32)
        x0, y0 = np.floor(pts.min(0) / g.dx).astype(int); x1, y1 = np.ceil(pts.max(0) / g.dx).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, g.n), min(y1, g.n)
        if x1 <= x0 or y1 <= y0: continue
        sub = centres.reshape(g.n, g.n, 2)[y0:y1, x0:x1].reshape(-1, 2)
        inside = MPath(pts).contains_points(sub).reshape(y1 - y0, x1 - x0)
        if not inside.any():                       # building smaller than one box: mark its centre box
            cx, cy = (pts.mean(0) / g.dx).astype(int)
            if 0 <= cx < g.n and 0 <= cy < g.n: inside = None; H[cy, cx] = max(H[cy, cx], h)
        if inside is not None:
            blk = H[y0:y1, x0:x1]; blk[inside] = np.maximum(blk[inside], h)
        if len(polys) < max_polys:
            polys.append(dict(h=round(h, 1), p=[[round(float(a), 1), round(float(b), 1)] for a, b in pts[:-1]]))
    print(f"    {len(j.get('elements', []))} buildings, {int((H > 0).sum())} grid columns covered")
    return H, polys, "OpenStreetMap contributors (ODbL)"


def fetch_places(g, plant_xy):
    print("[3/4] localities: OpenStreetMap place names ...")
    s, w, n, e = g.bbox()
    q = f'[out:json][timeout:60];(node["place"~"suburb|neighbourhood|village|hamlet|quarter|locality"]({s},{w},{n},{e}););out;'
    j = None
    for url in ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]:
        j = get_json(url, data=urllib.parse.urlencode({"data": q}).encode(), timeout=90, tries=2)
        if j: break
    places = []
    norm = lambda t: "".join(ch for ch in t.lower() if ch.isalpha())
    reported = {norm(r): r for r in REPORTED_AFFECTED}
    for el in (j or {}).get("elements", []):
        name = el.get("tags", {}).get("name:en") or el.get("tags", {}).get("name")
        if not name: continue
        x, y = g.to_xy(el["lat"], el["lon"])
        if not (0 <= x < g.n * g.dx and 0 <= y < g.n * g.dx): continue
        nm = norm(name)
        rep = any(k in nm or nm in k for k in reported if len(k) > 5)
        dist = math.hypot(x - plant_xy[0], y - plant_xy[1])
        places.append(dict(name=name, x=int(x / g.dx), y=int(y / g.dx), dist_m=round(dist),
                           reported=bool(rep), lat=el["lat"], lon=el["lon"]))
    places.sort(key=lambda p: p["dist_m"])
    print(f"    {len(places)} named places, {sum(p['reported'] for p in places)} match the reported-affected list")
    if not j: print("    Overpass unavailable -> no place names")
    return places[:40]


def fetch_weather():
    print("[4/4] weather: ERA5 hourly for 6-7 May 2020 ...")
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=%.4f&longitude=%.4f"
           "&start_date=2020-05-06&end_date=2020-05-07"
           "&hourly=temperature_2m,wind_speed_10m,wind_direction_10m,cloud_cover,precipitation"
           "&wind_speed_unit=ms&timezone=Asia%%2FKolkata") % (PLANT["lat"], PLANT["lon"])
    j = get_json(url)
    if not j or "hourly" not in j:
        print("    weather archive unavailable -> assumed calm night (1.5 m/s, stable)")
        hours = [dict(t=f"2020-05-07T{h:02d}:00", temp=27.0, wind=1.5, dir=225.0, cloud=20, rain=0.0) for h in range(0, 13)]
        return hours, "assumed (download failed)"
    h = j["hourly"]
    hours = [dict(t=t, temp=h["temperature_2m"][k], wind=h["wind_speed_10m"][k], dir=h["wind_direction_10m"][k],
                  cloud=h["cloud_cover"][k], rain=h["precipitation"][k])
             for k, t in enumerate(h["time"]) if t >= "2020-05-07T00:00" and t <= "2020-05-07T12:00"]
    for r in hours[:6]:
        print(f"    {r['t'][11:]} IST  {r['temp']:.1f} C  wind {r['wind']:.1f} m/s from {r['dir']:.0f} deg  cloud {r['cloud']}%")
    return hours, "ERA5 reanalysis via Open-Meteo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=float, default=5000, help="side of the square area in metres")
    ap.add_argument("--dx", type=float, default=10, help="box size in metres")
    ap.add_argument("--out", default=str(ROOT / "site" / "vizag.npz"))
    a = ap.parse_args()

    g = Grid(PLANT["lat"], PLANT["lon"], a.size, a.dx)
    print(f"Area: {a.size/1000:.1f} km x {a.size/1000:.1f} km around the plant, {g.n} x {g.n} boxes of {a.dx:.0f} m")
    elev, src_terrain = fetch_terrain(g)
    base = float(elev.min())
    terrain = elev - base
    buildings, polys, src_bld = fetch_buildings(g)
    px, py = g.to_xy(PLANT["lat"], PLANT["lon"])
    plant = dict(x=int(px / a.dx), y=int(py / a.dx), name=PLANT["name"], lat=PLANT["lat"], lon=PLANT["lon"])
    buildings[max(0, plant["y"] - 2):plant["y"] + 3, max(0, plant["x"] - 2):plant["x"] + 3] = 0  # tank yard open
    places = fetch_places(g, (px, py))
    weather, src_wx = fetch_weather()
    s, w, n, e = g.bbox()
    meta = dict(name="Visakhapatnam (LG Polymers), 7 May 2020", source="real",
                bbox=dict(s=s, w=w, n=n, e=e), lat0=PLANT["lat"], lon0=PLANT["lon"], base_elev_m=base,
                plant=plant, places=places, weather=weather, polygons=polys, facts=FACTS,
                data_sources=dict(terrain=src_terrain, buildings=src_bld, weather=src_wx,
                                  places="OpenStreetMap contributors (ODbL)"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, terrain=terrain.astype(np.float32), buildings=buildings.astype(np.float32),
                        dx=np.float32(a.dx), meta=json.dumps(meta))
    print(f"\nSaved {a.out}  (terrain relief {terrain.max():.0f} m, {len(polys)} building outlines, {len(places)} places)")


if __name__ == "__main__":
    main()
