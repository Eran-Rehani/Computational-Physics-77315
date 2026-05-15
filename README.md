# Computational Physics (77315)

Jupyter notebooks for the computational physics course (77315) at the Hebrew University of Jerusalem, Spring 2026. Covers numerical methods in Python applied to real physics problems — floating-point arithmetic, interpolation, numerical integration, root-finding, and stellar structure.

---

## Assignments

### HW 1 — Floating-Point Arithmetic & Numerical Stability
[`HW/HW_1_complete.ipynb`](HW/HW_1_complete.ipynb)

Computes powers of the golden mean $\phi = (\sqrt{5}-1)/2$ using two methods:
- **Forward recurrence**: direct computation $\phi^n$
- **Backward recurrence**: linear relation $\phi^{n-1} = \phi^{n+1}/\phi$ (numerically stable)

Demonstrates catastrophic cancellation in `float32` vs. `float64` and quantifies relative error as a function of $n$.

---

### HW 2 — Interpolation Methods
[`HW/HW2.ipynb`](HW/HW2.ipynb)

Implements and compares three interpolation schemes on a non-uniformly spaced grid of $\cos(x)$:

| Method | Order | Notes |
|--------|-------|-------|
| Linear | 1 | Two-point piecewise |
| Lagrange4 | 3 | Four-point polynomial |
| Hermit4 | 3 | Four-point with derivative matching |

Benchmarks accuracy at 101, 51, and 26 grid points to show convergence rates.

---

### HW 3 — Numerical Integration & Root-Finding: O₂ Quantum Energy Levels
[`HW/HW3.ipynb`](HW/HW3.ipynb)

Finds the vibrational energy levels of the O₂ molecule in the Bohr-Sommerfeld-Wilson quantization approximation using a Lennard-Jones potential:

$$V(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]$$

The quantization condition $\oint p\,dr = nh$ is evaluated numerically via adaptive quadrature (`scipy.integrate`), and energy levels are located by bisection/Brent root-finding.

---

### HW 4 — White Dwarf Stellar Structure
[`HW/HW_04.ipynb`](HW/HW_04.ipynb)

Solves the hydrostatic-equilibrium equations for a white dwarf using finite differences and Newton's method:

$$\frac{dP}{dr} = -\frac{G M(r) \rho(r)}{r^2}, \quad \frac{dM}{dr} = 4\pi r^2 \rho(r)$$

The electron degeneracy pressure obeys a relativistic equation of state. A Newton solver iterates to find the self-consistent stellar radius and mass from an initial guess.

---

## Setup

```bash
bash setup_and_run.sh   # creates .venv, installs deps, launches Jupyter
```

or manually:

```bash
pip install -r requirements.txt
jupyter notebook
```

## Dependencies

```
numpy  scipy  matplotlib  pandas  sympy  jupyter
```
