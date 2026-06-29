# HW 06 — 1D Lagrangian Hydrodynamics: Sod Shock Tube

## Overview

This assignment implements a staggered Lagrangian finite-difference scheme to solve the 1D Euler equations of compressible gas dynamics. The test problem is the **Sod shock tube**: a membrane separating two regions of different pressure is removed at $t=0$, and three wave structures emerge — a rarefaction fan, a contact discontinuity, and a shock. The simulation is run on four grid resolutions ($N = 34, 100, 334, 1000$) and compared against the analytic exact solution.

---

## Problem Statement

Evolve the ideal-gas Euler equations in 1D on the domain $[0, 16]$ from $t=0$ to $t=2.5$, with initial condition:

$$(\rho, u, p) = \begin{cases}(1.0,\; 0,\; 1.0) & x < 8 \\ (0.125,\; 0,\; 0.1) & x > 8\end{cases}$$

Rigid walls at both ends. Equation of state: $p = (\gamma-1)\rho e$ with $\gamma = 1.4$.

Compare numerically measured feature speeds (shock, contact, rarefaction head and tail) against the exact Riemann solution, and check energy conservation.

---

## The Physical Waves (Exact Solution)

The Sod problem produces three waves traveling at speeds determined by the Rankine-Hugoniot and isentropic relations:

| Feature | Speed (exact) | Description |
|---------|--------------|-------------|
| Rarefaction head | $-1.183$ | Left edge of the fan; moves left at $-c_L$ |
| Rarefaction tail | $-0.070$ | Right edge of the fan; barely moves |
| Contact discontinuity | $+0.927$ | Jump in $\rho$ only; no pressure or velocity jump |
| Shock | $+1.752$ | Compression wave; jumps in $\rho$, $p$, $u$ all |

The intermediate pressure $p_*$ is the root of $f_L(p_*) + f_R(p_*) + (u_R - u_L) = 0$, where each $f_K$ is a branch (rarefaction or shock) depending on whether $p_* < p_K$ or $p_* > p_K$. This is solved with `brentq`.

---

## The Numerical Scheme

### Staggered Lagrangian Grid

The grid is **staggered**: two interleaved sets of degrees of freedom move with the fluid (Lagrangian frame):

- **Cells** $(i-\frac{1}{2})$: hold $\rho$, $e$, $p$, $q$ (thermodynamic quantities, cell-centered).
- **Vertices** $(i)$: hold $x$, $u$, $a$ (position, velocity, acceleration).

**Cell mass is conserved exactly:** $m_{i-1/2} = \rho_{i-1/2}V_{i-1/2}$ is fixed. When vertices move, the cell volume changes and density is recomputed from $\rho = m/V$. This automatically satisfies mass conservation to machine precision — a key advantage of the Lagrangian approach.

**Initial setup:**
```python
x  = np.linspace(X_L, X_R, N + 1)   # N+1 vertices
xc = 0.5*(x[:-1] + x[1:])            # cell centres
rho = np.where(xc < X_DIAPH, RHO_L, RHO_R)
p   = np.where(xc < X_DIAPH, P_L, P_R)
e   = p / ((GAMMA - 1.0) * rho)
m   = rho * np.diff(x)               # fixed cell masses
```

### The Time-Step Algorithm (Leapfrog / Predictor-Corrector)

Each time step advances from $t^n$ to $t^{n+1}$. The scheme is a **2nd-order predictor-corrector** (sometimes called staggered leapfrog):

**Step 1 — Predictor velocity (half-step):**
$$u_i^{n+1/2} = u_i^n + \frac{\Delta t}{2}a_i^n$$
Walls enforced: $u_0 = u_N = 0$.

**Step 2 — New positions:**
$$x_i^{n+1} = x_i^n + \Delta t\,u_i^{n+1/2}$$

**Step 3 — New volumes and densities:**
$$V_{i-1/2}^{n+1} = x_i^{n+1} - x_{i-1}^{n+1}, \quad \rho_{i-1/2}^{n+1} = m_{i-1/2}/V_{i-1/2}^{n+1}$$

**Step 4 — Artificial viscosity** (compression only — no viscosity in expanding regions):
$$q_{i-1/2}^{n+1} = \begin{cases}\sigma\rho^{n+1}(u_{i-1}^{n+1/2} - u_i^{n+1/2})^2 & u_{i-1} > u_i \text{ (compression)}\\0 & \text{otherwise}\end{cases}$$

The parameter $\sigma = 3$ controls the shock-smearing width. Artificial viscosity is necessary: without it, the shock would produce unphysical oscillations (Gibbs phenomenon) in the finite-difference scheme. It converts kinetic energy into thermal energy over a few cells near the shock, mimicking the real dissipation in a thin shock layer.

**Step 5 — Energy update (algebraic implicit):**

The internal energy evolves by work done against pressure and viscosity:
$$e^{n+1} = \frac{e^n - \frac{\Delta V}{2m}(p^n + q^n + q^{n+1})}{1 + (\gamma-1)\frac{\Delta V}{2V^{n+1}}}$$

This form is **algebraically implicit** in $e^{n+1}$ and $p^{n+1} = (\gamma-1)\rho^{n+1}e^{n+1}$ simultaneously. For an ideal gas the system decouples analytically into a single formula, avoiding an iterative solve.

**Step 6 — New accelerations:**
$$a_i^{n+1} = -2\frac{(p+q)_{i+1/2}^{n+1} - (p+q)_{i-1/2}^{n+1}}{m_{i-1/2} + m_{i+1/2}}$$

Interior vertices only; walls held at $a=0$.

**Step 7 — Corrector velocity:**
$$u_i^{n+1} = u_i^{n+1/2} + \frac{\Delta t}{2}a_i^{n+1}$$

### Time Step Control

The time step must satisfy the **Courant-Friedrichs-Lewy (CFL) condition** — no information can travel more than one cell per step:

$$\Delta t \le \text{CFL} \cdot \min_i\frac{\Delta x_i}{c_i}$$

where $c_i = \sqrt{\gamma p_i/\rho_i}$ is the local sound speed. Additionally, a density-change limit prevents the density from changing by more than 5% per step:

$$\Delta t \le \frac{0.05\,\Delta x_i}{|\Delta u_i|}$$

A growth factor of 1.1 prevents the step from increasing too quickly after a refined region is passed.

```python
dt = CFL * np.min(dx / c)
du = np.abs(s['u'][1:] - s['u'][:-1])
dt = min(dt, MAX_DRHO * np.min(dx[moving] / du[moving]))
dt = min(dt, DT_GROWTH * dt_prev)
```

---

## Measuring Feature Speeds

Feature speeds are measured by locating each feature on two closely spaced snapshots (at $t=2.0$ and $t=2.2$) and computing $v = \Delta x/\Delta t$:

- **Shock:** rightmost crossing of the pressure level $\frac{1}{2}(p_* + p_R)$.
- **Contact:** crossing of the density level $\frac{1}{2}(\rho_{*L} + \rho_{*R})$ between the fan and the shock.
- **Rarefaction head:** crossing of $0.995\,\rho_L$ (just inside the undisturbed left region).
- **Rarefaction tail:** crossing of $1.005\,\rho_{*L}$ (just inside the star-left plateau).

Linear interpolation is used to locate the crossing precisely between grid points.

---

## Results

### Energy Conservation

| $N$ | $\Delta E/E_0$ |
|-----|---------------|
| 34 | $6.4\times10^{-5}$ |
| 100 | $2.1\times10^{-5}$ |
| 334 | $6.3\times10^{-6}$ |
| 1000 | $2.1\times10^{-6}$ |

The energy non-conservation scales as $N^{-1}$ (first-order), consistent with the first-order artificial viscosity scheme.

### Feature Speeds vs. Exact

| $N$ | $v_\text{shock}$ | $v_\text{contact}$ | $v_\text{head}$ | $v_\text{tail}$ |
|-----|-----------|-----------|----------|----------|
| 34 | 1.795 | 1.046 | −1.293 | −0.495 |
| 100 | 1.717 | 0.941 | −1.145 | +0.024 |
| 334 | 1.763 | 0.928 | −1.182 | −0.076 |
| 1000 | 1.753 | 0.927 | −1.176 | −0.074 |
| **exact** | **1.752** | **0.927** | **−1.183** | **−0.070** |

The shock, contact, and head converge to the exact values at $N=334$–$1000$. The **tail** is the hardest feature to resolve: it moves at only $-0.07$ cm/s and sits inside the diffuse rarefaction fan. It requires $N \ge 334$ cells to even locate correctly.

### Physical Interpretation

- The **rarefaction fan** is well-captured at all resolutions because it is a smooth expansion wave — finite differences handle smooth regions well.
- The **shock and contact** are smeared over $\sim\sigma = 3$ cell widths by the artificial viscosity; they sharpen as $N$ increases.
- The shock reaches $x \approx X_\text{diaph} + v_\text{shock}\cdot t \approx 12.4$ at $t=2.5$, well inside the domain $[0,16]$, so the rigid walls never interfere with the solution.

---

## Key Takeaways

- **Lagrangian grids move with the fluid**, so cell mass is conserved *exactly* (density is just $\rho=m/V$). This is the defining advantage over Eulerian (fixed-grid) schemes.
- **Staggered variables:** thermodynamic quantities ($\rho,e,p$) live at cell centers; kinematic quantities ($x,u,a$) live at vertices. This staggering gives natural second-order-accurate centered differences.
- **A predictor-corrector (leapfrog) time step** advances velocity at half-steps and positions/energy at full steps — second-order accurate in smooth regions.
- **Shocks need artificial viscosity.** Without it, finite-difference schemes produce spurious oscillations at discontinuities; the viscosity $q\propto\rho(\Delta u)^2$ (active only in compression) spreads each shock over a few cells and converts kinetic energy to heat.
- **The CFL condition** $\Delta t \le \text{CFL}\cdot\min(\Delta x/c)$ is the stability limit for explicit schemes: information must not cross more than one cell per step.
- **Always validate against an exact solution.** The Sod problem has an analytic Riemann solution, so measured wave speeds and energy drift can be checked quantitatively.
- **Resolution trade-offs:** smooth features (rarefaction fan) converge fast; discontinuities (shock, contact) stay smeared by viscosity and sharpen only as $N$ grows. Energy error falls $\sim N^{-1}$ — first-order, set by the viscosity at the shock.
