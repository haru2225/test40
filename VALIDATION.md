# Validation — 2026-09-24

Local environment: macOS arm64, Python 3.11, PyTorch 2.6.0, ASE 3.29.0, NumPy 1.26.4.

- `python -m pytest -q`: **7 passed**. Includes wrapped-score derivatives, symmetry,
  shared-oxygen mapping and mass conservation, atomic and CG end-to-end workflows,
  both phase conditions, loss reduction on a small fixed example, exact CPU restart.
- `bash -n run_test40.pbs`: passed.
- Actual reference preparation: glass 3000 atoms → 1000 SiO4 beads;
  crystal 192 atoms → 64 SiO4 beads. Glass has 8 oxygen sharing defects, explicitly
  recorded with the `--allow-sharing-defects` policy. Total masses:
  60084.3000 and 3845.3952 amu, respectively.
- Two CPU optimizer updates on these actual references, width 16, two layers:
  finite loss for both phases. Four reverse steps for each phase completed,
  generated extxyz/LAMMPS structures, evaluation JSON/PNG, and preserved bead masses
  across both export formats (including unequal masses in the glass template).

These short runs verify execution, not useful structure generation. Their outputs
are not trained production models and are not distributed as weights. In particular,
they do not establish correct glass RDF, crystal long-range order, or absence of overlaps.

GPU training, container build and PBS execution require the target infrastructure
and have not been run locally. `Singularity.def` uses the PyTorch 2.6.0 CUDA 12.4
runtime image. CI repeats CPU regression checks on Linux; see GitHub Actions for
the status of the uploaded commit.
