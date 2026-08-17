# Repository Guidelines

## Project Structure & Module Organization

This directory contains the 77315 Computational Physics final project:

- `nbody_solver.py` is the main Python implementation, including initial conditions, the Barnes-Hut octree, adaptive RK4 integration, diagnostics, fitting, and plotting.
- `nbody_solver.py --refit`, `--tolerance-fig`, and `--redraw` rebuild fitted tables and figures from saved `figures/run_V*.npz` states without rerunning the dynamics; `--rerun-v80=<tol>` repeats only the case supplying figures (a) to (c).
- `submission/` is the bundle handed in: `README.md`, `nbody_solver.py`, `report.pdf`, `requirements.txt` and `figures/`. It is its own git repository. Keep its copies byte-identical to the working copies in this directory.
- `report.tex` is the LaTeX report source; `figures/` contains generated plots, saved run states, and the generated table fragments (`figures/king_table.tex`, `figures/conv_table.tex`, `figures/theta_table.tex`) that `report.tex` pulls in with `\input`.
- `Final_project.pdf` is the assignment brief and the authoritative spec; `Final_project.md` is a working English translation and `Archive/PLAN.md` records the design decisions taken before implementation.
- `report.pdf`, `report.aux`, `report.log`, and similar files are build outputs. Do not edit generated files by hand.

## Build, Test, and Development Commands

Run commands from this directory:

```bash
python nbody_solver.py --test        # Fast self-checks (Kepler and Barnes-Hut comparisons)
python nbody_solver.py --quick       # Small simulation for debugging and plot validation
python nbody_solver.py --convergence # Energy drift vs RK tolerance -> figures/conv_table.tex
python nbody_solver.py --theta-table # Tree accuracy vs opening angle -> figures/theta_table.tex
python nbody_solver.py --tol=0.1332  # Production N=5000 run, Section 5 results
python nbody_solver.py --tol=133.2 --tag=eps01   # Same run at the literal eps=0.1 of the brief
pdflatex report.tex && pdflatex report.tex  # Build the report and resolve references
```

Install the packages in `submission/requirements.txt` in a project virtual environment when the active Python environment does not provide them.

## Coding Style & Naming Conventions

Use Python 4-space indentation, descriptive `snake_case` function and variable names, and uppercase names for physical constants and run-wide parameters. Keep units explicit in names or docstrings, matching the existing convention of kpc, M_sun, Gyr, and km/s. Preserve deterministic behavior by retaining the seeded random-number setup unless a change specifically requires new sampling behavior.

## Testing Guidelines

There is no external testing framework or coverage requirement. Add or update focused checks in the existing `run_tests()` path when changing numerical behavior. Run `--test` after solver changes and `--quick` when plots, output files, or integration flow are affected. Confirm that generated figures and `king_table.tex` are readable before rebuilding the report.

## Commit & Pull Request Guidelines

Recent history uses short, descriptive, imperative-style subjects, for example `add HW6 notebook` and `style: ...`. Follow that pattern: keep the subject concise and identify the affected component. Pull requests should explain the numerical or document change, list validation commands and relevant results, and include updated figures or report output when presentation changes. Avoid committing transient logs or unrelated generated artifacts.

## Numerical and Output Considerations

`Final_project.pdf` (Hebrew) is the authoritative specification; `Final_project.md` is a working translation and has been wrong before. Keep the project parameters and units consistent with the PDF. Production runs use `N=5000` and write outputs under `figures/`; do not launch one casually during iterative development. Review changes to softening, opening angle, timestep tolerance, or fit parameters against the report before merging.
