# AGENTS.md

Guidance for AI coding agents working in this repository.

## Repository Purpose

This repository stores Jupyter notebook exercises used in the ISBA lecture on AI/ML applications in accelerator science.

- 2024 notebooks are the baseline.
- 2026 work should update and refine the same exercises.

## Primary Working Style

This is a notebook-centric repository.

1. Treat each notebook as teaching material, not only executable code.
2. Prioritize clarity, reproducibility, and pedagogical structure.
3. Make incremental, explainable changes rather than full rewrites unless explicitly requested.
4. Assume Google Colaboratory is the default runtime unless the user states otherwise.

## What to Update for 2026

When asked to modernize content, focus on:

- Methodological updates (recent practice and caveats)
- Better evaluation and uncertainty reporting
- Reproducibility improvements (seed control, explicit preprocessing, clean outputs)
- Readability improvements in markdown explanations and figure labeling

## File and Change Boundaries

- Prefer changes inside existing notebooks unless the user requests new files.
- Keep data files in [data/](./data/) unchanged unless updates are explicitly requested.
- Do not delete original 2024 examples unless the user asks for removal.
- Keep [orbit_corr_rl.ipynb](./orbit_corr_rl.ipynb) as the 2026 RL example and preserve its NumPy/SciPy-first dependency policy unless explicitly changed by the user.
- Keep licensing and notice files intact:
  - [LICENSE.txt](./LICENSE.txt)
  - [NOTICE.txt](./NOTICE.txt)

## Notebook Editing Guidelines

- Maintain a top-to-bottom executable flow.
- Avoid hidden state dependencies between cells whenever possible.
- Use deterministic random seeds where relevant.
- Keep outputs concise to avoid bloated notebook diffs.
- Add short markdown rationale for any substantial methodological change.
- Write all notebook comments in English.
- Describe important calculation steps in Markdown cells, using LaTeX math notation when appropriate.
- As a default rule, edit the Jupytext-synchronized `.py` file rather than editing `.ipynb` directly.
- Keep notebook code and paired `.py` files synchronized with Jupytext.
- Prefer paths and setup steps that work in Google Colaboratory.

### Jupytext Policy

- Notebooks should be managed as paired `.ipynb` and `.py` files.
- Coding agents should primarily modify the paired `.py` files.
- Add a license declaration comment at the beginning of each notebook (via the paired `.py` file header and synced `.ipynb` first cell/comment).
- For newly created notebooks, follow the same license declaration style already used in the current `.ipynb` files.
- For notebooks newly created in 2026 (including [orbit_corr_rl.ipynb](./orbit_corr_rl.ipynb)), use `Copyright (C) 2026 ...` in the notebook license declaration.
- After modifying notebook content, sync pairs before finalizing changes.
- Recommended command:

```bash
jupytext --sync <notebook>.ipynb
```

### Git Commit Policy for Notebooks

- Strip notebook outputs before commit using `nbstripout`.
- Repository setup command:

```bash
nbstripout --install
```

- Optional manual cleanup command:

```bash
nbstripout *.ipynb
```

## Local Environment Policy

- Use Conda environment `isba` for local validation work.
- Keep the environment definition in [environment.yml](./environment.yml).
- Standard setup commands:

```bash
conda env create -f environment.yml
conda activate isba
```

## Validation Expectations

For notebook modifications:

1. Run edited notebooks (or at minimum edited cells) to ensure execution validity.
2. Confirm file/path references exist in this repository.
3. Preserve backward-compatible educational intent unless instructed otherwise.
4. Confirm edited notebooks are Colab-runnable and Jupytext-synced.
5. For behavior checks, it is acceptable to execute `.ipynb` and judge results from notebook outputs.

## Communication Style for Agents

- Explain *what* changed and *why* in concise terms.
- Distinguish between:
  - content modernization,
  - bug fixes,
  - presentation-only edits.
- If requirements are ambiguous, ask focused clarification questions before broad edits.
