# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Computational Physics (77315) course directory. Work is done in Jupyter notebooks using Python. The kernel name is `computational_physics_77315`.

## Environment Setup

```bash
# First-time setup: creates .venv, installs packages, registers kernel, launches Jupyter
bash setup_and_run.sh

# Manual activation after setup
source .venv/bin/activate
jupyter notebook
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`, `sympy` (see `requirements.txt`).

## Directory Structure

- `HW/` — Exercise instructions (`ex1_inst.pdf`, `ex2_inst.txt`) and starter notebooks (`HW_1.ipynb`). Course lecture PDFs (`course_2020_*.pdf`) are also here.
- `HW_1_complete.ipynb` — Completed homework 1 notebook (reference).
- `summary/` — Course summary materials.

## Homework Conventions

- Exercises provide exact function signatures to implement; match them precisely.
- `numpy` arrays are the standard data structure. Use `np.random.seed(12345)` for reproducibility when the instructions specify it.
- Interpolation exercises (ex2): implement `interp(x, y, x1, itype)` supporting `'Linear'`, `'Lagrange4'`, `'Hermit4'`.
