# ISBA Accelerator AI/ML Lecture Notebooks (2026 Edition)

This repository contains hands-on Jupyter notebook exercises for the ISBA accelerator school lecture on AI/ML applications to accelerator science.

The original materials were prepared for the 2024 lecture and are now being refreshed for the 2026 lecture with updated workflows, models, and discussion points.

## Scope

- Notebook-first examples for lecture and self-study
- Reproducible data-driven accelerator use cases
- Incremental updates from the 2024 baseline
- Primary execution target: Google Colaboratory

## Current Notebook Set

- [bayes_poisson_1.ipynb](./bayes_poisson_1.ipynb): Bayesian inference of the Poisson parameter $\lambda$ with a conjugate Gamma prior.
  - Updates the Gamma prior to the posterior using seeded random Poisson data.
  - Propagates the posterior uncertainty to the model function $P(k;\lambda)$ numerically (inverse-CDF sampling), giving the mean and the 68% credible interval.
  - Checks the numerical result against analytic solutions: the mean is the negative binomial distribution (posterior predictive), and the variance and quantiles are computed with Gamma integrals and the Lambert $W$ function.
- [gpr1.ipynb](./gpr1.ipynb): Gaussian process regression with the RBF kernel, implemented with NumPy/SciPy.
  - Builds up the posterior step by step: prior, one data point, and ten noisy samples of $f(x) = x\sin x$.
  - Draws functions from the posterior and compares the regression results for different length scales.
  - Optimizes the hyper-parameters ($\sigma_0$ and the length scale) by maximizing the log marginal likelihood.
- [gpopt1.ipynb](./gpopt1.ipynb): Gaussian process (Bayesian) optimization of a noisy 1D function $f(x) = x\sin x$, implemented with NumPy/SciPy.
  - Compares the expected improvement (EI) and upper confidence bound (UCB) acquisition functions step by step.
  - Shows how fitting the hyper-parameters from only a few points can trap the optimizer at a local maximum, and how bounds on the hyper-parameters avoid it.
  - Evaluates the success rate and the simple regret over 100 seeded runs.
- [qscan1.ipynb](./qscan1.ipynb): regression of the emittance and the Twiss parameters $(\alpha, \beta)$ from quadrupole scan data with a CNN (Keras).
  - Generates seeded random Gaussian beams and their quadrupole scan images (beam profile versus $1/f$) with a thin-lens quadrupole and a drift.
  - Trains a 2D CNN on the scan images with standardized targets and separate training, validation, and test sets.
  - Evaluates the RMSE and the relative error on the test set, and compares the predicted and true phase-space distributions.
- [tomography1.ipynb](./tomography1.ipynb): phase-space tomography with a CNN (Keras).
  - Generates seeded random two-component Gaussian beams and their tomography images (beam profiles for rotation angles from 0 to 178 degrees in normalized phase space).
  - Trains an encoder-decoder CNN that maps a tomography image to the 2D phase-space image, with the binary cross-entropy loss.
  - Evaluates the image error and the rms emittance and Twiss parameters calculated from the predicted images on the test set.
- [thyratron_grid_pca.ipynb](./thyratron_grid_pca.ipynb): abnormal-waveform detection of thyratron grid pulses of 46 new, aged, and failed tubes (data: [wfmdata.tar.gz](./wfmdata.tar.gz), extracted to `data/` automatically).
  - Reduces each 160-sample waveform to six principal-component scores (PCA) and separates new and failed tubes with a linear regression and a support vector regression.
  - Evaluates the separation on unseen tubes by leave-one-tube-out, and compares the output for the aged tubes with their operating hours.
- [thyratron_selfdischarge.ipynb](./thyratron_selfdischarge.ipynb): failure prognostics of a thyratron from its self-discharge counts (data: [thyratron_selfdischarge.csv](./thyratron_selfdischarge.csv)).
  - Infers the self-discharge rate with a Poisson likelihood and a conjugate Gamma prior, and derives the negative binomial predictive distribution of the hourly and cumulative counts (conditioning on the rate before integrating it out).
  - Detects the anomaly before the tube replacement by comparing the counts with the prediction from 24 hours before, and chooses the alarm threshold from the false alarms in a normal reference period.
- [orbit_corr_rl.ipynb](./orbit_corr_rl.ipynb): new in 2026; a reinforcement learning example of two-corrector orbit correction (see [New 2026 Notebooks](#new-2026-notebooks)).
- [diff_sim_1.ipynb](./diff_sim_1.ipynb): new in 2026; a differentiable-simulation example of Twiss matching with four thin-lens quadrupoles in JAX, comparing the number of simulation calls for Nelder-Mead and BFGS with finite-difference and automatic gradients (see [New 2026 Notebooks](#new-2026-notebooks)).

## New 2026 Notebooks

- [orbit_corr_rl.ipynb](./orbit_corr_rl.ipynb): reinforcement learning example for two-corrector orbit correction.
  - Implemented with NumPy/SciPy and Matplotlib; scikit-learn is used only for the Bayesian-optimization comparison.
  - Does not require `gymnasium`, `stable-baselines3`, or `torch`.
  - Trains a compact DDPG-style agent (linear actor, quadratic critic, replay buffer, target critic, Adam) from scratch, without prior knowledge, with checkpoint selection and early stopping.
  - Compares the learned policy with the nominal-model orbit feedback, the response-matrix correction, and Bayesian optimization, and shows the training variability over five seeds, including a warm start from the nominal feedback.
  - Includes a strong-quadrupole scenario in which the feedback from the model without the quadrupole fails, while RL still learns a working controller.
- [diff_sim_1.ipynb](./diff_sim_1.ipynb): differentiable-simulation example for Twiss (beam envelope) matching.
  - Beamline of five drifts and four thin-lens quadrupoles, written as pure JAX functions.
  - Matches the exit Twiss parameters from random but known entrance Twiss parameters by tuning the four quadrupole strengths.
  - Uses `jax.grad`, `jax.jacfwd`, `jax.hessian`, `jax.vmap`, and `jax.jit` for multi-start Adam optimization, Newton refinement, and sensitivity analysis of the loss to entrance Twiss errors.
  - Requires `jax` (preinstalled on Google Colaboratory; included in [environment.yml](./environment.yml) for local runs).

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
