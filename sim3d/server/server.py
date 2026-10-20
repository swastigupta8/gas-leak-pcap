"""
server.py — runs the simulation and streams it live to the 3D dashboard.

    browser  --(JSON controls: wind, leak, scenario ...)-->  server  --(ctypes)-->  libsim (CUDA or CPU)
    browser  <--(binary frames: ground map, 3D cloud, wind) + (JSON stats)--  server

Run:   python server/server.py            then open http://localhost:8000
Env:   SITE=site/vizag.npz  BACKEND=auto|cuda|cpu  PORT=8000
"""
import asyncio, json, math, os, time, threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from simcore import Site, Simulator, GASES, STABILITY, pasquill_class

ROOT = Path(__file__).resolve().parent.parent
SITE_PATH = Path(os.environ.get("SITE", ROOT / "site" / "vizag.npz"))
TARGET_FPS = 12
VOL_EVERY = 3                      # send the 3D cloud every N frames (it is the biggest message)
WIND3_EVERY = 24                   # send the 3D wind (Boxes view only) every N frames
CLOCK_START_MIN = 2 * 60 + 30      # replay starts 02:30 IST
LEAK_START_MIN = 2 * 60 + 42       # CCTV first shows vapour at 02:42 (HPC report)
# Events from the record, shown on the dashboard timeline (minutes after 02:30)
EVENTS = [dict(min=12, label="Vapour seen on CCTV", src="HPC report"),
          dict(min=24, label="Gas alarm sounds", src="HPC report"),
          dict(min=210, label="Evacuation under way", src="AIChE case study")]

# ------------------------------------------------------------------------------------------
# Build the site and the simulator
# ------------------------------------------------------------------------------------------
if SITE_PATH.exists():
    SITE = Site.load(SITE_PATH)
    print(f"[server] site: {SITE.meta['name']}  ({SITE.nx} x {SITE.ny} boxes of {SITE.dx} m)")
else:
    SITE = Site.synthetic(nx=int(os.environ.get("SYN_N", 160)), ny=int(os.environ.get("SYN_N", 160)), dx=10.0)
    print("[server] no site file found -> synthetic test valley (run data/prepare_site.py for Vizag)")

SIM = Simulator(SITE, backend=os.environ.get("BACKEND", "auto"))
print(f"[server] backend {SIM.backend}, grid {SIM.nx} x {SIM.ny} x {SIM.nz}, dt {SIM.dt:.3f} s")
GF = 2 if max(SIM.nx, SIM.ny) > 520 else 1              # ground map downsample
VF = max(1, round(max(SIM.nx, SIM.ny) / 128))           # 3D cloud downsample (horizontal)
WF = max(1, round(max(SIM.nx, SIM.ny) / 48))            # wind arrows downsample

BENCH = None
for p in [ROOT / "results" / "bench.json"]:
    if p.exists(): BENCH = json.loads(p.read_text())

# ------------------------------------------------------------------------------------------
# State shared with the browser
# ------------------------------------------------------------------------------------------
plant = SITE.meta["plant"]
state = dict(
    scenario="replay",                 # "replay" = the night of 7 May 2020, "whatif" = your own settings
    paused=os.environ.get("START_PAUSED", "1") == "1", warp=int(os.environ.get("WARP", 20)),             # simulation steps per frame
    kernel=1,                          # 1 = shared-memory gas kernel, 0 = naive
    gas="styrene", rate=10.0, height=12.0, leak_on=False,
    leak_x=plant["x"], leak_y=plant["y"],
    wind=2.0, dir=240.0, temp=27.0, stability="auto", decay=0.0,
)
commands = deque()
clients = {}
lock = threading.Lock()
status = dict(phase="warming up the air", progress=0.0)
places = [dict(p, now=0.0, peak=0.0, dose=0.0, first_aegl1=None, level=0) for p in SITE.meta.get("places", [])]
for p in places:
    p.setdefault("dist_m", round(math.hypot(p["x"] - plant["x"], p["y"] - plant["y"]) * SITE.dx))
clock = dict(sim_min=0.0)            # simulated minutes since 02:30
release = dict(started_min=None, released_kg=0.0)
wx_now = dict()


def ist(minutes_since_start):
    m = CLOCK_START_MIN + minutes_since_start
    return f"{int(m // 60) % 24:02d}:{int(m % 60):02d}"


def weather_at(minutes_since_start):
    """Replay: interpolate the hourly ERA5 record to the current simulated time."""
    hrs = SITE.meta.get("weather") or []
    if not hrs:
        return dict(wind=1.5, dir=225.0, temp=27.0, cloud=20, rain=0.0)
    t = (CLOCK_START_MIN + minutes_since_start) / 60.0      # hours after midnight on 7 May
    k = int(min(max(math.floor(t), 0), len(hrs) - 2)); f = min(max(t - k, 0.0), 1.0)
    a, b = hrs[k], hrs[k + 1]
    ang = math.radians(a["dir"]), math.radians(b["dir"])   # interpolate direction on the circle
    sx = (1 - f) * math.sin(ang[0]) + f * math.sin(ang[1]); cx = (1 - f) * math.cos(ang[0]) + f * math.cos(ang[1])
    return dict(wind=(1 - f) * a["wind"] + f * b["wind"], dir=math.degrees(math.atan2(sx, cx)) % 360,
                temp=(1 - f) * a["temp"] + f * b["temp"], cloud=(1 - f) * a["cloud"] + f * b["cloud"],
                rain=(1 - f) * (a.get("rain") or 0) + f * (b.get("rain") or 0))


def apply_weather():
    """Push the current weather (replay record or the user's what-if settings) into the simulator."""
    if state["scenario"] == "replay":
        w = weather_at(clock["sim_min"])
    else:
        w = dict(wind=state["wind"], dir=state["dir"], temp=state["temp"], cloud=20, rain=0.0)
    hour = (CLOCK_START_MIN + clock["sim_min"]) / 60.0 % 24
    night = hour < 6 or hour >= 18.5
    stab = state["stability"] if state["stability"] != "auto" else pasquill_class(w["wind"], w["cloud"], night, 30)
    decay = state["decay"] + (0.5 if w.get("rain", 0) > 0.5 else 0.0)    # rain washes some vapour out
    SIM.configure(wind=w["wind"], dir=w["dir"], temp=w["temp"], stability=stab, decay_per_hour=decay, gas=state["gas"])
    wx_now.update(w, stability=stab, stability_label=STABILITY[stab]["label"], night=night)


def apply_sources():
    if state["leak_on"]:
        SIM.set_sources([dict(x=state["leak_x"], y=state["leak_y"], height_m=state["height"], rate_kgs=state["rate"])])
    else:
        SIM.set_sources([])


def reset_run():
    SIM.clear_gas()
    clock["sim_min"] = 0.0
    release.update(started_min=None, released_kg=0.0)
    for p in places: p.update(now=0.0, peak=0.0, dose=0.0, first_aegl1=None, level=0)
    state["leak_on"] = False
    apply_weather(); apply_sources()


def tick():
    """One frame of work, in a worker thread: apply commands, step the sim, gather outputs."""
    with lock:
        while commands:
            c = commands.popleft()
            if c == "reset": reset_run()
            elif c == "weather": apply_weather()
            elif c == "sources": apply_sources()
            elif c == "reflow": SIM.reset_flow()
            elif c == "leak_now":                     # what-if runs start with the leak already on
                state["leak_on"] = True; apply_sources()
        if status["phase"] != "running":
            return None
        ms_air = ms_gas = 0.0; steps = 0
        if not state["paused"]:
            # replay: the leak starts by itself at 02:42
            if state["scenario"] == "replay" and release["started_min"] is None and \
                    CLOCK_START_MIN + clock["sim_min"] >= LEAK_START_MIN and not state.get("_manual_off"):
                state["leak_on"] = True; apply_sources()
            if state["leak_on"] and release["started_min"] is None:
                release["started_min"] = clock["sim_min"]
            steps = int(state["warp"])
            SIM.gas_mode = int(state["kernel"])
            ms_air, ms_gas = SIM.step(steps)
            dt_min = steps * SIM.dt / 60.0
            clock["sim_min"] += dt_min
            if state["leak_on"]: release["released_kg"] += state["rate"] * steps * SIM.dt
            if state["scenario"] == "replay" and int(clock["sim_min"]) % 2 == 0: apply_weather()
        else:
            dt_min = 0.0
        g = SIM.ground_ppm()
        out = dict(ground=g, steps=steps, ms_air=ms_air, ms_gas=ms_gas, dt_min=dt_min)
        out["wind"] = SIM.wind_slice(WF, 2)
        return out


def encode_log(ppm):
    """ppm -> byte, log scale: 0 = below 0.1 ppm, 1..255 = 0.1 .. 10,000 ppm (same as the GPU)."""
    with np.errstate(divide="ignore"):
        t = (np.log10(np.maximum(ppm, 1e-9)) + 1.0) / 5.0
    b = np.where(ppm < 0.1, 0, 1 + np.clip(t * 254, 0, 254)).astype(np.uint8)
    return b


def summarise(out):
    g = out["ground"]; gas = GASES[state["gas"]]; dx = SIM.dx
    zones = []
    py, px = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    for key in ("aegl1", "aegl2", "aegl3"):
        m = g >= gas[key]
        reach = 0.0
        if m.any():
            reach = float(np.sqrt(((px[m] - state["leak_x"]) ** 2 + (py[m] - state["leak_y"]) ** 2).max()) * dx)
        zones.append(dict(key=key, ppm=gas[key], area_km2=float(m.sum()) * dx * dx / 1e6, reach_m=reach))
    for p in places:
        x0, x1 = max(0, p["x"] - 1), min(SIM.nx, p["x"] + 2); y0, y1 = max(0, p["y"] - 1), min(SIM.ny, p["y"] + 2)
        v = float(g[y0:y1, x0:x1].max()) if x1 > x0 and y1 > y0 else 0.0
        p["now"] = v; p["peak"] = max(p["peak"], v); p["dose"] += v * out["dt_min"]
        lvl = 3 if p["peak"] >= gas["aegl3"] else 2 if p["peak"] >= gas["aegl2"] else 1 if p["peak"] >= gas["aegl1"] else 0
        if lvl >= 1 and p["first_aegl1"] is None: p["first_aegl1"] = ist(clock["sim_min"])
        p["level"] = lvl
    cells = SIM.nx * SIM.ny * SIM.nz
    mlups = cells / (out["ms_air"] * 1e-3) / 1e6 if out["ms_air"] > 0 else 0.0
    since = None if release["started_min"] is None else clock["sim_min"] - release["started_min"]
    return dict(t="stats", clock=ist(clock["sim_min"]), since_min=since, sim_min=clock["sim_min"],
                ms_air=out["ms_air"], ms_gas=out["ms_gas"], steps=out["steps"], mlups=mlups,
                dt=SIM.dt, peak=float(g.max()), zones=zones, released_t=release["released_kg"] / 1000.0,
                places=[dict(name=p["name"], dist_m=p.get("dist_m"), reported=p.get("reported", False),
                             now=p["now"], peak=p["peak"], dose=p["dose"], first=p["first_aegl1"], level=p["level"])
                        for p in places],
                weather=wx_now, state={k: v for k, v in state.items() if not k.startswith("_")},
                backend=SIM.backend, status=status)


async def broadcast_json(msg):
    data = json.dumps(msg)
    for ws in list(clients):
        try: await ws.send_text(data)
        except Exception: clients.pop(ws, None)


async def sim_loop():
    # 1. spin up the air flow (wind settles around hills and buildings) before anything is released
    apply_weather(); apply_sources()
    SIM.gas_mode = -1
    total = int(os.environ.get("SPINUP", 1200))
    done = 0
    while done < total:
        n = min(100, total - done)
        await asyncio.to_thread(SIM.step, n)
        done += n
        status["progress"] = done / total
        await broadcast_json(dict(t="status", status=status))
    SIM.clear_gas(); clock["sim_min"] = 0.0
    pre = int(os.environ.get("TEST_PRERUN", 0))          # testing only: run the leak before showing it
    if pre:
        state["leak_on"] = True; apply_sources(); SIM.gas_mode = 0
        release["started_min"] = 0.0
        for _ in range(pre // 100):
            await asyncio.to_thread(SIM.step, 100)
            clock["sim_min"] += 100 * SIM.dt / 60; release["released_kg"] += state["rate"] * 100 * SIM.dt
    status.update(phase="running", progress=1.0)
    frame = 0; t_last = time.perf_counter(); fps = 0.0
    while True:
        t0 = time.perf_counter()
        if not clients:
            await asyncio.sleep(0.2); continue
        out = await asyncio.to_thread(tick)
        if out is None:
            await asyncio.sleep(0.1); continue
        stats = summarise(out)
        now = time.perf_counter(); fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-3); t_last = now
        stats["fps"] = fps
        stats["speed_x"] = (out["steps"] * SIM.dt) * fps if out["steps"] else 0.0
        g = out["ground"]
        gb = encode_log(g[::GF, ::GF] if GF == 1 else g.reshape(g.shape[0] // GF, GF, g.shape[1] // GF, GF).max(axis=(1, 3)))
        wind = np.clip(np.round(out["wind"] * 10), -127, 127).astype(np.int8)
        msgs = [b"\x01" + np.array(gb.shape, np.uint16).tobytes() + gb.tobytes(),
                b"\x03" + np.array(wind.shape[:2], np.uint16).tobytes() + wind.tobytes()]
        if frame % VOL_EVERY == 0:
            vol = await asyncio.to_thread(SIM.volume_bytes, VF, VF, 1)
            msgs.append(b"\x02" + np.array(vol.shape, np.uint16).tobytes() + vol.tobytes())
        wind3 = None
        if frame % WIND3_EVERY == 0 and any(c.get("view") == "boxes" for c in clients.values()):
            w3 = await asyncio.to_thread(SIM.wind_volume, VF, VF, 2)
            wind3 = b"\x07" + np.array(w3.shape[1:], np.uint16).tobytes() + w3.tobytes()
        for ws, info in list(clients.items()):
            if not info["ready"] and now - info["sent"] < 3.0:
                continue                                   # browser still drawing the last frame
            info["ready"], info["sent"] = False, now
            try:
                await ws.send_text(json.dumps(stats))
                for m in msgs: await ws.send_bytes(m)
                if wind3 is not None and info.get("view") == "boxes": await ws.send_bytes(wind3)
            except Exception:
                clients.pop(ws, None)
        frame += 1
        await asyncio.sleep(max(0.0, 1.0 / TARGET_FPS - (time.perf_counter() - t0)))


def hello():
    m = SITE.meta
    polys = m.get("polygons") or []
    if not polys:      # synthetic site: make simple square outlines from the building grid
        ys, xs = np.nonzero(SITE.buildings)
        polys = [dict(h=float(SITE.buildings[y, x]), p=[[x * SITE.dx, y * SITE.dx], [(x + 1) * SITE.dx, y * SITE.dx],
                 [(x + 1) * SITE.dx, (y + 1) * SITE.dx], [x * SITE.dx, (y + 1) * SITE.dx]]) for y, x in zip(ys[:6000], xs[:6000])]
    return dict(t="hello", name=m["name"], source=m.get("source"), bbox=m.get("bbox"), nx=SIM.nx, ny=SIM.ny,
                nz=SIM.nz, dx=SIM.dx, plant=m["plant"], polygons=polys, facts=m.get("facts"),
                data_sources=m.get("data_sources"), weather_series=m.get("weather"),
                gases={k: v for k, v in GASES.items()}, stability={k: v["label"] for k, v in STABILITY.items()},
                bench=BENCH, backend=SIM.backend, ground_factor=GF, wind_factor=WF, vol_factor=VF,
                leak_start=ist(LEAK_START_MIN - CLOCK_START_MIN), status=status,
                places_xy=[dict(x=p["x"], y=p["y"]) for p in places], events=EVENTS,
                timeline_min=240, clock_start="02:30")


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(sim_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(ROOT / "web" / "index.html", headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text(json.dumps(hello()))
    bld = np.clip(np.round(SITE.buildings), 0, 255).astype(np.uint8)              # building heights, metres
    await ws.send_bytes(b"\x05" + np.array(bld.shape, np.uint16).tobytes() + bld.tobytes())
    terr = np.clip(np.round(SITE.terrain * 10), 0, 65535).astype(np.uint16)      # decimetres
    await ws.send_bytes(b"\x04" + np.array(terr.shape, np.uint16).tobytes() + terr.tobytes())
    sc = SIM.solid_coarse(VF, VF)                                                 # boxes for the Boxes view
    await ws.send_bytes(b"\x06" + np.array(sc.shape, np.uint16).tobytes() + sc.tobytes())
    clients[ws] = {"ready": True, "sent": 0.0, "view": "map"}
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            k = msg.get("t")
            if k == "ack":
                if ws in clients: clients[ws]["ready"] = True
            elif k == "set":
                changed = set()
                for key, val in msg.get("values", {}).items():
                    if key in state and not key.startswith("_"):
                        state[key] = type(state[key])(val) if state[key] is not None else val
                        changed.add(key)
                if changed & {"wind", "dir", "temp", "stability", "decay", "gas", "scenario"}: commands.append("weather")
                if changed & {"rate", "height", "leak_on", "leak_x", "leak_y", "gas"}: commands.append("sources")
                if "leak_on" in changed and not state["leak_on"]: state["_manual_off"] = True
                if "scenario" in changed: commands.append("reset")
            elif k == "view":
                if ws in clients:
                    clients[ws]["view"] = "boxes" if msg.get("v") == "boxes" else "map"
                    if clients[ws]["view"] == "boxes":          # send the wind straight away
                        w3 = await asyncio.to_thread(SIM.wind_volume, VF, VF, 2)
                        await ws.send_bytes(b"\x07" + np.array(w3.shape[1:], np.uint16).tobytes() + w3.tobytes())
            elif k == "reset":
                state.pop("_manual_off", None); commands.append("reset")
            elif k == "whatif":                       # "Run this night" in the What if drawer
                for key, val in msg.get("values", {}).items():
                    if key in state and not key.startswith("_"):
                        state[key] = type(state[key])(val)
                state["scenario"] = "whatif"; state["paused"] = False; state.pop("_manual_off", None)
                commands.extend(["reset", "weather", "leak_now"])
            elif k == "replay":                       # "Back to 2020"
                state["scenario"] = "replay"; state.pop("_manual_off", None)
                state.update(rate=10.0, height=12.0, gas="styrene", stability="auto", decay=0.0,
                             leak_x=plant["x"], leak_y=plant["y"])
                commands.extend(["reset", "weather"])
    except WebSocketDisconnect:
        pass
    finally:
        clients.pop(ws, None)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), log_level="warning")
