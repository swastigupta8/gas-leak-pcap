# sim3d: the 3D Vizag 2020 simulator

Originally written by Dhruvishi Amin as a single Colab notebook; the files here are that notebook's source cells, unchanged, plus a `Makefile`.

It replays the 7 May 2020 styrene vapour release at LG Polymers, Visakhapatnam:

- **Air flow:** Lattice Boltzmann method, D3Q19 (19 velocity directions per cell), with a Smagorinsky turbulence model and buoyancy for the heavy vapour. Terrain and buildings are solid cells.
- **Gas:** concentration transported by that flow.
- **Data** (`data/prepare_site.py`, needs internet): Copernicus 90 m terrain, OpenStreetMap buildings and place names, ERA5 hourly weather for the night.
- **One source, two targets:** `core/compat.h` lets the same kernels compile with `nvcc` for the GPU or with `g++` + OpenMP for the CPU, which makes the GPU-vs-CPU speed-up a like-for-like comparison.
- **MPI** (`core/mpi_sim.cu`): the map is split into strips, with halo rows exchanged every step; the result is checked against a single-process run.
- **Dashboard** (`server/server.py` + `web/index.html`): 3D view with live what-if controls.

## Run

```bash
make            # CPU builds (any Linux machine / WSL)
make cuda       # GPU builds (needs nvcc)
python data/prepare_site.py --size 5000 --dx 10    # download the site data
python server/server.py                             # dashboard on http://localhost:8000
```

The whole sequence, with figures, is part B of `../notebooks/colab_run_all.ipynb`.

How it relates to the 2D core: the 2D model in `../core2d` is the clean, fully benchmarked stencil study the synopsis promises. This 3D model shows the same parallel techniques on a physically realistic problem.
