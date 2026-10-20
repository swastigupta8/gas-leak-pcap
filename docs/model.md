# The 2D model

This is the maths behind `core2d/`. Every number in this file comes from the code or from the tests in `tests/test_correctness.py`.

## 1. Equation

The gas concentration $C(x, y, t)$ in mg/m³ is averaged over a ground layer of depth $H$ (10 m by default). It is carried by a uniform wind $(u, v)$, spread by turbulent eddies with diffusivity $D$, and fed by a leak $S$:

$$
\frac{\partial C}{\partial t} + u\frac{\partial C}{\partial x} + v\frac{\partial C}{\partial y} = D\left(\frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2}\right) + S
$$

| Symbol | Meaning | Default |
|---|---|---|
| $h$ | cell size | 1 m (the synopsis's "one thread per metre") |
| $D$ | eddy diffusivity | 10 m²/s |
| $u, v$ | wind towards east, north | set from speed and direction |
| $H$ | mixing-layer depth | 10 m (dashboard) |
| $S$ | leak of $Q$ kg/s into one cell | $Q \cdot 10^6 / (h^2 H)$ mg/m³ per second |

## 2. Discretisation: the 5-point stencil

The scheme is explicit in time (forward Euler). Diffusion uses central differences. Advection uses first-order upwinding: each cell takes gas from the neighbour the wind is coming from, which never creates new peaks or troughs. With $u^+ = \max(u, 0)$ and $u^- = \max(-u, 0)$ (likewise for $v$), every term reduces to a weighted sum of a cell and its four neighbours:

$$
C^{n+1}_{i,j} = w_0 C_{i,j} + w_E C_{i+1,j} + w_W C_{i-1,j} + w_N C_{i,j+1} + w_S C_{i,j-1}
$$

$$
w_E = \frac{\Delta t}{h}\left(\frac{D}{h} + u^-\right),\quad
w_W = \frac{\Delta t}{h}\left(\frac{D}{h} + u^+\right),\quad
w_N = \frac{\Delta t}{h}\left(\frac{D}{h} + v^-\right),\quad
w_S = \frac{\Delta t}{h}\left(\frac{D}{h} + v^+\right)
$$

$$
w_0 = 1 - (w_E + w_W + w_N + w_S) = 1 - \Delta t\left(\frac{4D}{h^2} + \frac{|u| + |v|}{h}\right)
$$

The weights are computed once per run on the host (`g2_make_weights` in `common.c`). The kernel is then nothing but this weighted sum. The weights add up to exactly 1, so a uniform field stays uniform and no mass is created or lost inside the map.

## 3. Stability: the time step is not free

Every weight must be non-negative. Otherwise a cell's new value can overshoot, concentrations go negative, and the run blows up within a few steps. $w_0 \ge 0$ gives the limit

$$
\Delta t \le \frac{1}{4D/h^2 + (|u| + |v|)/h}
$$

The code uses 0.9 × this limit and refuses anything larger (`tests::test_unstable_time_step_is_refused`).

**What this means for the synopsis target.** With $h = 1$ m, $D = 10$ m²/s and a 3–4 m/s wind, $\Delta t = 0.0205$ s, so **49 steps simulate one second**. The synopsis asked for "30 iterations per second". At 30 steps per second the simulation would run at 0.6× real time: slower than the gas itself. The meaningful target is the **real-time factor**, simulated seconds per wall-clock second. A responder who wants a 30-minute forecast within one minute needs 30×, which is about 1,500 steps per second on the 4000 × 4000 grid.

The limit scales with $h^2$, which is the central resolution trade-off:

| Cell size $h$ | Δt | Cells per km² | Work per simulated second (relative) |
|---|---|---|---|
| 1 m | 0.020 s | 1,000,000 | 1 |
| 2 m | 0.075 s | 250,000 | 1/15 |
| 5 m | 0.375 s | 40,000 | 1/460 |

Halving the cell size costs about 8–16× more work: 4× more cells, and up to 4× more steps.

## 4. Accuracy, measured

`tests/test_correctness.py` compares the simulation with the exact solution for a puff released at $(x_0, y_0)$:

$$
C(x, y, t) = \frac{M}{4\pi D t}\exp\left(-\frac{(x - x_0 - ut)^2 + (y - y_0 - vt)^2}{4Dt}\right)
$$

Results on a 400 m square map, 20 s of simulation:

| Case | h = 2 m | h = 1 m | h = 0.5 m |
|---|---|---|---|
| Diffusion only: relative L2 error | 0.06% | 0.01% | 0.01% |
| With 3 m/s wind: relative L2 error | 8.2% | 4.5% | 2.4% |

In all six runs the total mass stayed within 0.006% of the amount released (single-precision rounding). Diffusion is near-exact. With wind, the error halves every time $h$ halves. That is the signature of a first-order scheme. Upwinding adds **numerical diffusion** of about $\tfrac{|u| h}{2}$, which smears the plume a little wider and lower than reality: at $h = 1$ m the peak is about 6% low. A second-order advection scheme (e.g. Lax–Wendroff with a flux limiter) would reduce this, at the cost of a wider stencil.

## 5. What the model leaves out

These assumptions decide what the output can and cannot be used for:

- **2D, well-mixed layer.** No vertical structure. Dense gases such as styrene vapour hug the ground at night; the 3D simulator in `sim3d/` models that, this model does not.
- **Flat ground, uniform wind.** Buildings and terrain channel real plumes; `sim3d/` includes them.
- **Constant $D$.** Real turbulent mixing depends on atmospheric stability and changes over the night.
- **Edges absorb.** Ghost cells are fixed at zero, so gas leaving the map is lost. The diffusion term slightly over-removes gas near edges where wind blows outward, so a plume should stay well inside the map.

It is a teaching and benchmarking model for parallel stencil computation, not an operational tool. Real responders use validated models such as EPA/NOAA ALOHA.
