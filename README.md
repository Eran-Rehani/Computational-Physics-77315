# Computational Physics 77315

Course workspace for "Introduction to Computational Physics (77315)", Spring Semester 2026.

## Contents

- `lectures_summary.tex`: main LaTeX source for the combined lecture summary document.
- `transcripts/`: lecture transcripts and raw notes used to build the summaries.
- `Lecture_Summaries/`: per-lecture summary drafts and supporting notes.
- `HW/`: homework notebooks and assignment instructions.
- `summary/`: reference course summary PDFs.
- `Literature/`: books and external reference material.
- `requirements.txt`: Python dependencies for notebooks and numerical work.
- `setup_and_run.sh`: creates a virtual environment, installs dependencies, and launches Jupyter.

## Typical workflow

### Work on the notes

Edit `lectures_summary.tex` and compile it with your preferred LaTeX toolchain, for example:

```bash
latexmk -pdf lectures_summary.tex
```

The repository is configured to ignore common LaTeX build artifacts such as `.aux`, `.log`, `.fls`, and `lectures_summary.pdf`.

### Work on notebooks

Set up the Python environment and launch Jupyter:

```bash
bash setup_and_run.sh
```

Installed packages currently include Jupyter, NumPy, SciPy, Matplotlib, Pandas, and SymPy.

## Notes

- Source files and reference materials are intended to stay tracked.
- Temporary notebook checkpoints, virtual environments, assistant metadata, and generated LaTeX artifacts are ignored.
