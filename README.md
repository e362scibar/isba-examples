# ISBA Accelerator AI/ML Lecture Notebooks (2026 Edition)

This repository contains hands-on Jupyter notebook exercises for the ISBA accelerator school lecture on AI/ML applications to accelerator science.

The original materials were prepared for the 2024 lecture and are now being refreshed for the 2026 lecture with updated workflows, models, and discussion points.

## Scope

- Notebook-first examples for lecture and self-study
- Reproducible data-driven accelerator use cases
- Incremental updates from the 2024 baseline
- Primary execution target: Google Colaboratory

## Current Notebook Set

- [bayes_poisson_1.ipynb](./bayes_poisson_1.ipynb)
- [gpr1.ipynb](./gpr1.ipynb)
- [gpopt1.ipynb](./gpopt1.ipynb)
- [qscan1.ipynb](./qscan1.ipynb)
- [tomography1.ipynb](./tomography1.ipynb)
- [thyratron_grid_pca.ipynb](./thyratron_grid_pca.ipynb)
- [thyratron_selfdischarge.ipynb](./thyratron_selfdischarge.ipynb)

## Data and Auxiliary Files

- Notebook data files are stored in [data/](./data/).
- Additional lecture assets include:
  - [thyratron_selfdischarge.csv](./thyratron_selfdischarge.csv)
  - [wfmdata.tar.gz](./wfmdata.tar.gz)

## 2026 Refresh Policy

When updating the notebooks for 2026:

1. Preserve the pedagogical flow of the 2024 material.
2. Keep examples executable on Google Colaboratory as the default environment.
3. Prefer explicit, reproducible preprocessing and evaluation steps.
4. Clarify where a method is historical baseline vs. current best practice.
5. Document major conceptual updates directly in notebook markdown cells.
6. Write notebook comments in English.
7. Explain key calculation processes in Markdown cells, using LaTeX equations where useful.

## Suggested Update Themes for 2026

- Better uncertainty communication (calibration, confidence intervals, predictive distributions)
- Evaluation rigor (cross-validation strategy, leakage checks, baseline comparisons)
- Practical model governance for research environments (reproducibility and traceability)
- Modernized optimization workflow (clear objective definitions and robust stopping criteria)
- Notebook ergonomics (clear sections, deterministic seeds, concise plotting)

## Quick Start

1. Open notebooks in Google Colaboratory whenever possible.
2. Verify required Python packages are installed in the Colab runtime.
3. Run notebooks top-to-bottom after checking paths and mounted storage.
4. If needed, unpack [wfmdata.tar.gz](./wfmdata.tar.gz) before running waveform-related sections.

## Local Validation Environment (Conda)

For local notebook checks, use the managed Conda environment:

```bash
conda env create -f environment.yml
conda activate isba
```

Environment definition is tracked in [environment.yml](./environment.yml).

## Notebook Source Management (Jupytext)

All notebooks should be synchronized with paired `.py` files via Jupytext.

- Keep each `.ipynb` paired with a `.py` representation.
- When editing content, keep `.ipynb` and `.py` in sync before commit.
- Recommended command:

```bash
jupytext --sync <notebook>.ipynb
```

## Git Commit Hygiene (`nbstripout`)

Notebook output should be stripped before committing to Git.

- Install `nbstripout` in your environment.
- Enable it in this repository:

```bash
nbstripout --install
```

- If needed, clean existing notebooks before commit:

```bash
nbstripout *.ipynb
```

## License and Notice

This project is distributed under GPLv3 (or later). See:

- [LICENSE.txt](./LICENSE.txt)
- [NOTICE.txt](./NOTICE.txt)
