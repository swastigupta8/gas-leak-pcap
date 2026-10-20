"""
simcore.py — drives the C++/CUDA simulator from Python and handles all the real-world units.

The simulator itself only knows "lattice units" (1 box = 1, 1 step = 1). This file converts:
    metres, seconds, m/s, kg/s, °C, cloud cover  ->  lattice numbers
and back again (ppm, km², minutes) for the dashboard.
"""
import ctypes, math, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ------------------------------------------------------------------------------------------
# Gas and weather science
# ------------------------------------------------------------------------------------------
GASES = {
    # molar mass g/mol; AEGL thresholds in ppm for 60-minute exposure (US EPA, interim for styrene)
    "styrene": dict(name="Styrene", M=104.15, aegl1=20, aegl2=130, aegl3=1100, idlh=700),
    "ammonia": dict(name="Ammonia", M=17.03, aegl1=30, aegl2=160, aegl3=1100, idlh=300),
    "chlorine": dict(name="Chlorine", M=70.9, aegl1=0.5, aegl2=2.0, aegl3=20, idlh=10),
}
M_AIR = 28.97
R_GAS = 8.314

# Pasquill–Gifford stability classes: A (very unstable, sunny) ... F (very stable, calm clear night)
#   p        : wind-profile exponent  u(z) = u10 (z/10)^p   (urban values, US EPA)
#   kbg_h/v  : background turbulent mixing not resolved by the grid (m²/s)
#   kv       : multiplier on vertical turbulent mixing (stable air suppresses vertical motion)
STABILITY = {
    "A": dict(p=0.15, kbg_h=6.0, kbg_v=3.0, kv=2.0,  label="Very unstable (strong sun)"),
    "B": dict(p=0.15, kbg_h=4.0, kbg_v=2.0, kv=1.6,  label="Unstable"),
    "C": dict(p=0.20, kbg_h=2.5, kbg_v=1.2, kv=1.3,  label="Slightly unstable"),
    "D": dict(p=0.25, kbg_h=1.5, kbg_v=0.6, kv=1.0,  label="Neutral (overcast or windy)"),
    "E": dict(p=0.30, kbg_h=0.8, kbg_v=0.15, kv=0.4, label="Slightly stable (night)"),
    "F": dict(p=0.30, kbg_h=0.4, kbg_v=0.05, kv=0.15, label="Stable (calm, clear night)"),
}


def pasquill_class(wind_ms, cloud_pct, is_night, sun_elev_deg=45.0):
    """Turner's method, simplified: classic lookup from wind speed, cloud cover and day/night."""
    w = wind_ms
    if is_night:
        if cloud_pct >= 50:
            return "E" if w < 3 else "D"
        return "F" if w < 2 else ("E" if w < 3 else "D")
    strong = sun_elev_deg > 50 and cloud_pct < 50
    slight = sun_elev_deg < 25 or cloud_pct > 75
    if w < 2:  return "A" if strong else ("B" if not slight else "B")
    if w < 3:  return "A" if strong else ("B" if not slight else "C")
    if w < 5:  return "B" if strong else ("C" if not slight else "C")
    if w < 6:  return "C" if strong else ("D")
    return "C" if strong else "D"


# ------------------------------------------------------------------------------------------
# Site: terrain + buildings + places, either from the prepared real data or synthetic
# ------------------------------------------------------------------------------------------
class Site:
    def __init__(self, terrain, buildings, dx, meta):
        self.terrain = terrain.astype(np.float32)        # metres above lowest point, (ny, nx)
        self.buildings = buildings.astype(np.float32)    # building height above terrain, (ny, nx)
        self.dx = float(dx)
        self.meta = meta                                 # dict: name, bbox, source, places, weather...

    @property
    def ny(self): return self.terrain.shape[0]
    @property
    def nx(self): return self.terrain.shape[1]

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        return Site(d["terrain"], d["buildings"], float(d["dx"]), meta)

    @staticmethod
    def synthetic(nx=160, ny=160, dx=10.0, seed=1):
        """A made-up valley with hills and a village grid, for testing without internet."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        X, Y = xx / nx, yy / ny
        terrain = (90 * np.exp(-((X - 0.15) ** 2 + (Y - 0.8) ** 2) / 0.03)      # hill NW
                   + 70 * np.exp(-((X - 0.85) ** 2 + (Y - 0.75) ** 2) / 0.02)   # hill NE
                   + 25 * Y + 6 * np.sin(6 * X) * np.cos(5 * Y))
        terrain -= terrain.min()
        buildings = np.zeros_like(terrain)
        for _ in range(int(nx * ny / 60)):
            bx, by = rng.integers(5, nx - 5), rng.integers(5, ny - 5)
            if terrain[by, bx] < 40:
                w, h = rng.integers(1, 3), rng.integers(1, 3)
                buildings[by:by + h, bx:bx + w] = rng.choice([6, 9, 12, 15])
        cx, cy = nx // 2, ny // 3
        buildings[cy - 2:cy + 3, cx - 2:cx + 3] = 0      # clear the plant yard
        meta = dict(name="Synthetic test valley", source="synthetic", lat0=17.7553, lon0=83.2089,
                    plant=dict(x=cx, y=cy, name="Plant"),
                    places=[dict(name=f"Village {k+1}", x=int(np.clip(cx + fx * nx, 2, nx - 3)), y=int(np.clip(cy + fy * ny, 2, ny - 3)))
                            for k, (fx, fy) in enumerate([(.16, .06), (-.12, .19), (.25, -.09), (.03, .34), (-.28, -.06)])],
                    weather=None, polygons=[])
        return Site(terrain, buildings, dx, meta)


# ------------------------------------------------------------------------------------------
# The simulator wrapper
# ------------------------------------------------------------------------------------------
class Simulator:
    def __init__(self, site, nz=None, backend="auto", height_m=None):
        self.site = site
        self.lib, self.backend = self._load(backend)
        dx = site.dx
        top = float((site.terrain + site.buildings).max())
        if height_m is None:
            height_m = top + 80.0
        self.nz = nz or int(min(64, max(8, math.ceil(height_m / dx))))
        self.nx, self.ny = site.nx, site.ny
        self.dx = dx
        L = self.lib
        if L.sim_create(self.nx, self.ny, self.nz) != 0:
            raise RuntimeError("sim_create failed")
        self._build_geometry()
        self.sources = []          # list of dict(x, y, height_m, rate_kgs)
        self.gas = GASES["styrene"]
        self.sim_time = 0.0
        self.gas_mode = 0
        self.weather = dict(wind=2.0, dir=270.0, temp=27.0, stability="F")
        self.configure(**self.weather)
        L.sim_init_flow()

    # ---- loading the compiled library ----
    def _load(self, backend):
        names = {"cuda": "libsim_cuda.so", "cpu": "libsim_cpu.so"}
        order = ["cuda", "cpu"] if backend == "auto" else [backend]
        last = None
        for b in order:
            p = ROOT / "build" / names[b]
            if not p.exists():
                last = f"{p} not built"; continue
            try:
                lib = ctypes.CDLL(str(p))
            except OSError as e:
                last = str(e); continue
            self._sig(lib)
            if b == "cuda":
                # quick probe: a CUDA library without a GPU fails at allocation time
                pass
            return lib, b
        raise RuntimeError(f"no simulator library could be loaded: {last}")

    @staticmethod
    def _sig(L):
        f, i, P = ctypes.c_float, ctypes.c_int, ctypes.c_void_p
        L.sim_backend.restype = ctypes.c_char_p
        L.sim_create.argtypes, L.sim_create.restype = [i, i, i], i
        L.sim_set_geometry.argtypes = [P, P, P]
        L.sim_set_flow.argtypes = [f] * 12
        L.sim_set_gas.argtypes = [f] * 7
        L.sim_set_sources.argtypes = [i, P, P]
        L.sim_step.argtypes = [i, i, P]
        L.sim_ground.argtypes = [P]
        L.sim_volume.argtypes = [P, i, i, i]
        L.sim_wind.argtypes = [P, i, i]
        L.sim_wind_volume.argtypes = [P, i, i, i, f]
        L.sim_get_C.argtypes = [P]
        L.sim_get_u.argtypes = [P, P, P]
        L.sim_set_threads.argtypes = [i]

    # ---- terrain and buildings -> solid boxes ----
    def _build_geometry(self):
        s, dx, nz = self.site, self.dx, self.nz
        zc = (np.arange(nz, dtype=np.float32) + 0.5) * dx          # height of each box centre
        terr = s.terrain[None, :, :]
        total = (s.terrain + s.buildings)[None, :, :]
        solid = (zc[:, None, None] < total).astype(np.uint8)       # (nz, ny, nx)
        solid[-1] = 0
        self.ground = np.clip(np.ceil(s.terrain / dx - 0.5), 0, nz - 2).astype(np.int32)   # first air box over terrain
        self.breath = np.clip(np.ceil((s.terrain + s.buildings) / dx - 0.5), 0, nz - 2).astype(np.int32)
        self.solid = np.ascontiguousarray(solid)
        self.lib.sim_set_geometry(self.solid.ctypes.data, self.ground.ctypes.data, self.breath.ctypes.data)

    # ---- weather -> air + gas parameters ----
    def configure(self, wind=None, dir=None, temp=None, stability=None, decay_per_hour=None, gas=None):
        w = self.weather
        if wind is not None: w["wind"] = float(max(0.3, wind))
        if dir is not None: w["dir"] = float(dir) % 360
        if temp is not None: w["temp"] = float(temp)
        if stability is not None: w["stability"] = stability
        if decay_per_hour is not None: w["decay"] = float(decay_per_hour)
        if gas is not None: self.gas = GASES[gas]
        st = STABILITY[w["stability"]]
        dx = self.dx
        H = self.nz * dx
        u_top = w["wind"] * max(1.0, (H / 10.0)) ** st["p"]
        # Time step: the fastest wind (top of the profile) moves 0.15 boxes per step, and dt never
        # exceeds 0.05*dx seconds. If the wind changes a lot, pick a new dt and restart the air flow
        # (the gas cloud is kept).
        dt_new = min(0.05 * dx, 0.15 * dx / max(u_top, 4.0))
        if not hasattr(self, "dt"):
            self.dt = dt_new
        elif dt_new < 0.8 * self.dt or dt_new > 1.25 * self.dt:
            self.dt = dt_new
            self._retimed = True
        dt = self.dt
        lat = lambda v_ms: v_ms * dt / dx             # m/s -> lattice
        latK = lambda k: k * dt / dx ** 2             # m²/s -> lattice
        th = math.radians(w["dir"])                   # wind FROM this direction
        dirx, diry = -math.sin(th), -math.cos(th)     # unit vector it blows TO (x east, y north)
        beta = self.gas["M"] / M_AIR - 1.0            # how much heavier than air (per unit mole fraction)
        g_lb = 9.81 * dt * dt / dx
        self.lib.sim_set_flow(0.506, 0.17, lat(w["wind"]), st["p"], dirx, diry, 10.0 / dx,
                              0.16, max(6.0, 0.06 * min(self.nx, self.ny)), 0.08, g_lb * beta, 0.22)
        decay = w.get("decay", 0.0) / 3600.0 * dt
        self.lib.sim_set_gas(1 / 0.7, latK(st["kbg_h"]), latK(st["kbg_v"]), st["kv"], 0.07, 0.2, decay)
        self.u_lb_top = lat(u_top)
        self._push_sources()
        if getattr(self, "_retimed", False):
            self._retimed = False
            self.lib.sim_init_flow()

    # ---- leaks ----
    def vapour_density(self):
        T = self.weather["temp"] + 273.15
        return 101325.0 * self.gas["M"] * 1e-3 / (R_GAS * T)   # kg/m³ of pure vapour

    def set_sources(self, sources):
        self.sources = list(sources)
        self._push_sources()

    def _push_sources(self):
        xyz, rates = [], []
        rho_v = self.vapour_density()
        for s in self.sources:
            x, y = int(s["x"]), int(s["y"])
            if not (0 <= x < self.nx and 0 <= y < self.ny):
                continue
            z = int(self.breath[y, x] + max(0, round(s.get("height_m", 0) / self.dx)))
            z = min(z, self.nz - 2)
            vol_rate = s["rate_kgs"] / rho_v                       # m³/s of pure vapour
            ppm_per_step = vol_rate * self.dt / self.dx ** 3 * 1e6
            xyz += [x, y, z]; rates.append(ppm_per_step)
        a = np.array(xyz or [0, 0, 0], np.int32); r = np.array(rates or [0], np.float32)
        self.lib.sim_set_sources(len(rates), a.ctypes.data, r.ctypes.data)

    # ---- running ----
    def step(self, n):
        ms = np.zeros(2, np.float32)
        self.lib.sim_step(int(n), int(self.gas_mode), ms.ctypes.data)
        self.sim_time += n * self.dt
        return float(ms[0]), float(ms[1])

    def reset_flow(self):
        self.lib.sim_init_flow()

    def clear_gas(self):
        self.lib.sim_clear_gas(); self.sim_time = 0.0

    # ---- outputs ----
    def ground_ppm(self):
        out = np.empty((self.ny, self.nx), np.float32)
        self.lib.sim_ground(out.ctypes.data)
        return out

    def volume_bytes(self, fx, fy, fz):
        vx, vy, vz = -(-self.nx // fx), -(-self.ny // fy), -(-self.nz // fz)
        out = np.empty((vz, vy, vx), np.uint8)
        self.lib.sim_volume(out.ctypes.data, fx, fy, fz)
        return out

    def wind_slice(self, f, zoff=2):
        wx, wy = -(-self.nx // f), -(-self.ny // f)
        out = np.empty((wy, wx, 2), np.float32)
        self.lib.sim_wind(out.ctypes.data, f, zoff)
        return out * (self.dx / self.dt)                          # lattice -> m/s

    def wind_volume(self, fx, fy, fz):
        """3D wind shrunk by (fx, fy, fz): int8 array (3, vz, vy, vx) in 0.1 m/s (east, north, up)."""
        vx, vy, vz = -(-self.nx // fx), -(-self.ny // fy), -(-self.nz // fz)
        out = np.empty((3, vz, vy, vx), np.int8)
        self.lib.sim_wind_volume(out.ctypes.data, fx, fy, fz, 10.0 * self.dx / self.dt)
        return out

    def solid_coarse(self, fx, fy):
        """Solid boxes sampled at the centre of each (fx, fy, 1) block: 0 air, 1 ground, 2 building."""
        vx, vy = -(-self.nx // fx), -(-self.ny // fy)
        xs = np.minimum(np.arange(vx) * fx + fx // 2, self.nx - 1)
        ys = np.minimum(np.arange(vy) * fy + fy // 2, self.ny - 1)
        sol = self.solid[:, ys][:, :, xs]
        zc = (np.arange(self.nz, dtype=np.float32) + 0.5) * self.dx
        terr = self.site.terrain[np.ix_(ys, xs)]
        ground = zc[:, None, None] < terr[None]
        return np.where(sol == 0, 0, np.where(ground, 1, 2)).astype(np.uint8)

    def full_C(self):
        out = np.empty((self.nz, self.ny, self.nx), np.float32)
        self.lib.sim_get_C(out.ctypes.data); return out

    def full_u(self):
        a = [np.empty((self.nz, self.ny, self.nx), np.float32) for _ in range(3)]
        self.lib.sim_get_u(a[0].ctypes.data, a[1].ctypes.data, a[2].ctypes.data)
        return [v * (self.dx / self.dt) for v in a]
