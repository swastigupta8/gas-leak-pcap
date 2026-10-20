"""
server.py - live 2D dashboard for the core2d simulator.

    browser --(JSON: leak position, wind, leak rate, speed)--> server --(ctypes)--> libgas2d
    browser <--(binary heat-map frame + JSON stats, ~30 per second)-- server

Run (from the repository root, after `make -C core2d` and optionally `make -C core2d cuda`):

    pip install -r dashboard/requirements.txt
    python dashboard/server.py            then open http://localhost:8000

Environment variables:
    BACKEND = auto | cuda | cpu   (auto tries the GPU library first)
    GRID    = cells per side      (default 2000 -> a 2 km x 2 km map at 1 m resolution)
    PORT    = 8000

Model assumptions the numbers depend on (see docs/model.md):
  * The 2D model tracks the concentration averaged over a ground layer MIX_HEIGHT metres deep.
    A leak of Q kg/s therefore raises its cell by Q / (h^2 * MIX_HEIGHT) kg/m^3 per second.
  * ppm assumes 25 degC and 1 atm: ppm = mg/m^3 * 24.45 / molar mass.
  * Colour bands are the US EPA 60-minute AEGL thresholds for styrene (the Vizag 2020 chemical).
"""
import asyncio
import ctypes
import json
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "core2d" / "build"
STATIC = Path(__file__).resolve().parent / "static"

GRID = int(os.environ.get("GRID", "2000"))
FRAME = min(500, GRID)  # pixels per side sent to the browser
H = 1.0  # cell size, m
D = 10.0  # eddy diffusivity, m^2/s (neutral conditions, urban surface)
MIX_HEIGHT = 10.0  # depth of the ground layer the gas is mixed through, m
FPS = 30

CHEMICAL = dict(name="Styrene", molar_mass=104.15, aegl1=20.0, aegl2=130.0, aegl3=1100.0)
PPM_PER_MG_M3 = 24.45 / CHEMICAL["molar_mass"]
LOG_MAX = 4.0  # frames encode log10(1 + ppm) in 0..LOG_MAX as one byte per pixel


def load_library(backend):
    names = {"cuda": "libgas2d_cuda.so", "cpu": "libgas2d_cpu.so"}
    order = ["cuda", "cpu"] if backend == "auto" else [backend]
    errors = []
    for b in order:
        path = BUILD / names[b]
        if not path.exists():
            errors.append(f"{path.name} not built")
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as e:  # e.g. the CUDA library on a machine without a GPU driver
            errors.append(f"{path.name}: {e}")
            continue
        lib.g2s_create.restype = ctypes.c_void_p
        lib.g2s_create.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.g2s_set_params.restype = ctypes.c_float
        lib.g2s_set_params.argtypes = [ctypes.c_void_p] + [ctypes.c_float] * 5
        lib.g2s_set_source.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.g2s_clear.argtypes = [ctypes.c_void_p]
        lib.g2s_step.restype = ctypes.c_double
        lib.g2s_step.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.g2s_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
        lib.g2s_backend.restype = ctypes.c_char_p
        return lib
    raise SystemExit("no simulator library could be loaded: " + "; ".join(errors) + " (run `make -C core2d`)")


class Simulation:
    """One shared simulation; every connected browser sees and controls the same run."""

    def __init__(self, lib):
        self.lib = lib
        self.sim = lib.g2s_create(GRID, GRID)
        if not self.sim:
            raise SystemExit(f"could not allocate a {GRID} x {GRID} grid")
        self.backend = lib.g2s_backend().decode()
        self.wind_speed, self.wind_from = 3.0, 225.0  # m/s, degrees (meteorological: direction it blows FROM)
        self.rate_kg_s = 10.0
        self.leak = None  # (i, j) cell or None
        self.speed = 10.0  # simulated seconds per wall-clock second requested by the user
        self.sim_time = 0.0
        self.ms_per_step = 0.0
        self.frame = np.zeros((FRAME, FRAME), np.float32)
        # Controls arrive on the event loop while stepping runs in a worker thread; the lock stops
        # a parameter change from landing in the middle of a batch of steps.
        self.lock = threading.Lock()
        self.apply()

    def apply(self):
        # Meteorological convention: wind FROM 225 deg (south-west) blows TOWARDS the north-east.
        rad = math.radians(self.wind_from)
        u, v = -self.wind_speed * math.sin(rad), -self.wind_speed * math.cos(rad)
        self.dt = self.lib.g2s_set_params(self.sim, H, D, u, v, 0.0)
        if self.leak:
            # kg/s -> mg/m^3 added to the leak cell per step
            per_step = self.rate_kg_s * 1e6 / (H * H * MIX_HEIGHT) * self.dt
            self.lib.g2s_set_source(self.sim, self.leak[0], self.leak[1], per_step)
        else:
            self.lib.g2s_set_source(self.sim, -1, -1, 0.0)

    def control(self, msg):
        with self.lock:
            self._control(msg)

    def _control(self, msg):
        kind = msg.get("type")
        if kind == "leak":
            self.leak = (int(np.clip(msg["x"], 0, 0.9999) * GRID), int(np.clip(msg["y"], 0, 0.9999) * GRID))
        elif kind == "stop":
            self.leak = None
        elif kind == "wind":
            self.wind_speed = float(np.clip(msg["speed"], 0, 15))
            self.wind_from = float(msg["from"]) % 360
        elif kind == "rate":
            self.rate_kg_s = float(np.clip(msg["kg_s"], 0.01, 500))
        elif kind == "speed":
            self.speed = float(np.clip(msg["x"], 1, 300))
        elif kind == "reset":
            self.lib.g2s_clear(self.sim)
            self.sim_time = 0.0
            self.leak = None
        self.apply()

    def advance(self, wall_seconds):
        """Run enough steps to cover speed x wall_seconds of simulated time, within the frame budget."""
        want = max(1, round(self.speed * wall_seconds / self.dt))
        # If the hardware cannot keep up, run what fits in ~70% of the frame time instead of
        # falling behind; the stats then show the achievable speed honestly.
        if self.ms_per_step > 0:
            want = min(want, max(1, int(0.7 * 1000 * wall_seconds / self.ms_per_step)))
        with self.lock:
            ms = self.lib.g2s_step(self.sim, want)
            self.ms_per_step = ms / want
            self.sim_time += want * self.dt
            self.lib.g2s_frame(self.sim, self.frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), FRAME, FRAME)
        return want

    def encoded_frame(self):
        ppm = self.frame * PPM_PER_MG_M3
        code = np.clip(np.log10(1.0 + ppm) / LOG_MAX * 255.0, 0, 255).astype(np.uint8)
        return code.tobytes()

    def stats(self, steps, wall_ms):
        ppm = self.frame * PPM_PER_MG_M3
        px_km2 = (GRID * H / FRAME) ** 2 / 1e6
        # Pixels are max-pooled, so these areas are slight over-estimates: the safe direction.
        return dict(
            type="stats", backend=self.backend, grid=GRID, cell_m=H, frame=FRAME, sim_time=self.sim_time,
            dt=self.dt, steps=steps, ms_per_step=self.ms_per_step, wall_ms=wall_ms,
            realtime=(self.dt * 1000 / self.ms_per_step) if self.ms_per_step > 0 else 0.0,
            mlups=(GRID * GRID / (self.ms_per_step * 1e3)) if self.ms_per_step > 0 else 0.0,
            peak_ppm=float(ppm.max()), leak=self.leak, rate=self.rate_kg_s, wind_speed=self.wind_speed,
            wind_from=self.wind_from, speed=self.speed, chemical=CHEMICAL, log_max=LOG_MAX,
            area_km2={k: float((ppm >= CHEMICAL[k]).sum() * px_km2) for k in ("aegl1", "aegl2", "aegl3")},
        )


SIM: Simulation | None = None
clients: set[WebSocket] = set()


@asynccontextmanager
async def lifespan(_app):
    """Create the simulation once at startup and start the frame loop."""
    global SIM
    SIM = Simulation(load_library(os.environ.get("BACKEND", "auto")))
    print(f"backend: {SIM.backend}  grid: {GRID} x {GRID} at {H} m  dt: {SIM.dt:.4f} s", flush=True)
    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    clients.add(sock)
    try:
        while True:
            SIM.control(json.loads(await sock.receive_text()))
    except (WebSocketDisconnect, KeyError, ValueError, TypeError):
        pass
    finally:
        clients.discard(sock)


async def loop():
    """Fixed-rate loop: step, render, broadcast. Simulation work runs in a worker thread
    (ctypes releases the GIL), so the server stays responsive to controls meanwhile."""
    period = 1.0 / FPS
    while True:
        t0 = time.perf_counter()
        if clients:
            steps = await asyncio.to_thread(SIM.advance, period)
            frame, stats = SIM.encoded_frame(), SIM.stats(steps, (time.perf_counter() - t0) * 1e3)
            for c in list(clients):
                try:
                    await c.send_text(json.dumps(stats))
                    await c.send_bytes(frame)
                except Exception:
                    clients.discard(c)
        await asyncio.sleep(max(0.0, period - (time.perf_counter() - t0)))




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="warning")
