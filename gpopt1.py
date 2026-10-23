# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: isba
#     language: python
#     name: python3
# ---

# %% [markdown] id="JEPVI5dsKphB"
# # Gaussian process optimizer
#
# Copyright (C) 2026 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.


# %% executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1792389072212, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="mN7BxlXiKphD"
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
# %matplotlib inline

# %% executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1792389072230, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="ZMRz5rkWKphE"
# Parameters
plt.rcParams['font.size'] = 16
seed = 2026  # random seed for reproducibility
z95 = scipy.stats.norm.ppf(0.975)  # half width of the 95% interval in units of the standard deviation
sigma_0 = 2.0  # prior standard deviation (default hyper-parameter)
lscale = 1.0  # length scale of the kernel (default hyper-parameter)
sigma_e = 0.5  # measurement error
kappa = 1.96  # exploration factor of UCB
niter = 10  # number of acquisitions after the first measurement
xmin = 0.
xmax = 10.
nx = 201
x = np.linspace(xmin, xmax, nx)
ylim = (-10., 20.)  # y range of the plots
tol_opt = 0.3  # tolerance to regard a point as the global optimum


# %% [markdown] id="qh0BfjXgKphF"
# ## Function definition
# - True function: $y = x \sin{x}$
# - Each measurement has a Gaussian error with the standard deviation $\sigma_e$.
# - The global maximum is at $x \approx 7.98$ and a local maximum is at $x \approx 2.03$.

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1792389072237, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="UEhWUigtKphF"
# True function and data taking function
def true_func(x):
    return x * np.sin(x)

def take_data(x, rng):
    return true_func(x) + rng.normal(0., sigma_e)

x_fine = np.linspace(xmin, xmax, 100001)
x_opt = x_fine[np.argmax(true_func(x_fine))]  # global maximum
local_range = x_fine < 4.
x_local = x_fine[local_range][np.argmax(true_func(x_fine[local_range]))]  # local maximum
print(f'global maximum: x = {x_opt:.3f}, y = {true_func(x_opt):.3f}')
print(f'local maximum:  x = {x_local:.3f}, y = {true_func(x_local):.3f}')


# %% [markdown] id="CCBW7642KphF"
# ## Gaussian process regression
# Same as `gpr1.ipynb`: RBF kernel $K(x_a,x_b) = \sigma_0^2 \exp\left(-\frac{|x_a-x_b|^2}{2\lambda^2}\right)$,
# and the linear systems with $\boldsymbol\Sigma_\mathcal{D}$ are solved by the Cholesky decomposition.
# Only the diagonal of the posterior covariance matrix (the variance at each $x$) is needed for the acquisition functions:
# $$
# \mu_f(x) = \boldsymbol\kappa_\mathcal{D}^\top \boldsymbol\Sigma_\mathcal{D}^{-1} \boldsymbol{y}_\mathcal{D},
# \qquad
# \sigma_f^2(x) = \sigma_0^2 - \boldsymbol\kappa_\mathcal{D}^\top \boldsymbol\Sigma_\mathcal{D}^{-1} \boldsymbol\kappa_\mathcal{D}
# $$

# %% executionInfo={"elapsed": 133, "status": "ok", "timestamp": 1792389072373, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="FG5SW7uvKphG"
# Gaussian process regression
def rbf_kernel(xa, xb, sigma0, lscale=1.0):
    """RBF kernel matrix K[i, j] = sigma0^2 exp(-|xa[i] - xb[j]|^2 / (2 lscale^2))."""
    dx = np.subtract.outer(xa, xb)
    return sigma0**2 * np.exp(-0.5*(dx/lscale)**2)


def gpr(x, xd, yd, err, sig0, lscale=1.0):
    """Posterior mean and standard deviation of the GP at x, given the data (xd, yd)."""
    Sigma_d = rbf_kernel(xd, xd, sig0, lscale) + np.eye(len(xd))*err**2
    kd = rbf_kernel(xd, x, sig0, lscale)
    chol = scipy.linalg.cho_factor(Sigma_d)
    mean = kd.T @ scipy.linalg.cho_solve(chol, yd)
    var = sig0**2 - np.sum(kd * scipy.linalg.cho_solve(chol, kd), axis=0)
    return mean, np.sqrt(np.clip(var, 0., None))  # rounding errors can make tiny negative variances


# %% [markdown]
# ## Hyper-parameter optimization
# The hyper-parameters $(\sigma_0, \lambda)$ can be updated at every iteration by maximizing the log marginal likelihood
# (see `gpr1.ipynb`):
# $$
# \ln{p(\mathcal{D}|\mathrm{X}_\mathcal{D})} =
# -\frac{1}{2} \boldsymbol{y}_\mathcal{D}^\top \boldsymbol\Sigma_\mathcal{D}^{-1} \boldsymbol{y}_\mathcal{D}
# -\frac{1}{2} \ln{|\boldsymbol\Sigma_\mathcal{D}|}
# -\frac{N_\mathcal{D}}{2} \ln{2\pi}
# $$
# - With only a few data points, the marginal likelihood is too flat to determine the hyper-parameters.
#   Therefore, the default values are used until `n_min_fit` points are available, and the values are kept within bounds.
#   The bounds act as prior knowledge on the hyper-parameters; two choices are compared at the end of this notebook.
# - The main examples below use the fixed default values ($\sigma_0=2$, $\lambda=1$) as in the lecture slides.
#   Note that $\sigma_0=2$ is smaller than the amplitude of the true function (about 8),
#   so the credible interval far from the data is too narrow.
#   The effect of the hyper-parameter optimization is shown at the end of this notebook.

# %%
n_min_fit = 3  # minimum number of data points to fit the hyper-parameters
bounds_wide = [(0.5, 20.), (0.3, 5.)]  # loose bounds of (sigma_0, length scale)
bounds_narrow = [(0.5, 20.), (0.3, 1.5)]  # length scale limited by the expected smoothness of the function


def log_marginal_likelihood(xd, yd, sig0, lscale, err):
    """Log marginal likelihood ln p(D | X_D) of the GP with the RBF kernel."""
    Sigma_d = rbf_kernel(xd, xd, sig0, lscale) + np.eye(len(xd))*err**2
    chol = scipy.linalg.cho_factor(Sigma_d)
    log_det = 2. * np.sum(np.log(np.diag(chol[0])))
    return -0.5 * yd @ scipy.linalg.cho_solve(chol, yd) - 0.5 * log_det - 0.5 * len(xd) * np.log(2.*np.pi)


def fit_hyperparameters(xd, yd, err, init, bounds=bounds_wide):
    """Maximize the log marginal likelihood over (ln sigma_0, ln length scale) within the bounds."""
    if len(xd) < n_min_fit:
        return init
    log_init = np.clip(np.log(init), *np.log(bounds).T)
    res = scipy.optimize.minimize(
        lambda theta: -log_marginal_likelihood(xd, yd, *np.exp(theta), err),
        log_init, method='L-BFGS-B', bounds=np.log(bounds),
    )
    return tuple(np.exp(res.x))


# %% [markdown] id="s7ZyMaUKKphG"
# ## Expected improvement (EI)
# $$
# \alpha_\mathrm{EI}(x) = \int_{y^*}^\infty{(y-y^*)\, p(y|\boldsymbol{y}_\mathcal{D},\mathrm{X}_\mathcal{D},x)\, dy}
# = (\mu(x) - y^*) \Phi(t) + \sigma(x) \varphi(t)
# $$
# - $t = \frac{\mu(x) - y^*}{\sigma(x)}$
# - $\Phi(t)$: cumulative distribution function of the standard normal distribution
# - $\varphi(t)$: probability density function of the standard normal distribution
# - EI is the *expected amount* of improvement over the best value $y^*$, not the probability of improvement.
# - $y^*$ is the best value seen so far, $y^* = y_\mathrm{max} = \max(\boldsymbol{y}_\mathcal{D})$, as in the lecture slides.
# - Caveat: the measured values contain errors, so $\max(\boldsymbol{y}_\mathcal{D})$ is biased upward by lucky fluctuations.
#   Noise-aware variants exist, e.g. the maximum of the posterior mean at the measured points,
#   $y^* = \max_i \mu(x_{\mathcal{D},i})$, or the noisy EI in BoTorch.
#   Such a choice changes the balance between exploration and exploitation;
#   the simple posterior-mean variant is included in the comparison at the end of this notebook.

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1792389072377, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="RGosRp0HKphH"
# Expected improvement
def EI(mu, sig, y_best):
    """Expected improvement over y_best for the posterior mean mu and standard deviation sig."""
    sig = np.maximum(sig, 1e-12)  # avoid division by zero
    t = (mu - y_best) / sig
    return (mu - y_best) * scipy.stats.norm.cdf(t) + sig * scipy.stats.norm.pdf(t)


# %% [markdown] id="LXvM5nQJKphH"
# ## Upper confidence bound (UCB)
# $$
# \alpha_\mathrm{UCB}(x) = \mu(x) + \kappa\sigma(x)
# $$
# - A larger $\kappa$ gives more weight to the uncertainty (exploration),
#   and a smaller $\kappa$ gives more weight to the mean (exploitation).
# - Here, $\kappa = 1.96$, i.e. the upper edge of the 95% credible interval.

# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1792389072377, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="AFFh0d-_KphH"
# Upper confidence bound
def UCB(mu, sig, kappa=kappa):
    """Upper confidence bound mu + kappa * sig."""
    return mu + kappa*sig


# %% [markdown]
# ## Optimization loop
# 1. Take the first data point at the center of the range.
# 2. Compute the GP posterior from the data (optionally after fitting the hyper-parameters).
# 3. Evaluate the acquisition function on the grid and take the next data point at its maximum.
# 4. Repeat 2–3 `niter` times.
#
# The maximum of the acquisition function is searched on a grid of 201 points.
# This is possible only in low dimensions; for many parameters, a numerical optimizer
# (with multiple starting points) is used to maximize the acquisition function.

# %%
def acquisition(name, mu, sig, y_d, mu_d, incumbent='observed'):
    """Evaluate the acquisition function 'EI' or 'UCB'.

    For EI, the best value y* is the maximum of the measured values (incumbent='observed')
    or of the posterior mean at the measured points (incumbent='posterior').
    """
    if name == 'EI':
        return EI(mu, sig, np.max(y_d if incumbent == 'observed' else mu_d))
    return UCB(mu, sig)


def run_optimization(name, rng, niter=niter, fit_hyper=False, bounds=bounds_wide, incumbent='observed'):
    """Run the GP optimization and return the data and the history of each iteration."""
    x_d = np.array([x[nx//2]])
    y_d = np.array([take_data(x_d[0], rng)])
    hyper = (sigma_0, lscale)
    history = []
    for _ in range(niter):
        if fit_hyper:
            hyper = fit_hyperparameters(x_d, y_d, sigma_e, hyper, bounds)
        mean, sig = gpr(x, x_d, y_d, sigma_e, *hyper)
        mu_d, _ = gpr(x_d, x_d, y_d, sigma_e, *hyper)  # posterior mean at the measured points
        acq = acquisition(name, mean, sig, y_d, mu_d, incumbent)
        x_next = x[np.argmax(acq)]
        history.append(dict(x_d=x_d, y_d=y_d, mean=mean, sig=sig, acq=acq, x_next=x_next, hyper=hyper))
        x_d = np.append(x_d, x_next)
        y_d = np.append(y_d, take_data(x_next, rng))
    return x_d, y_d, history


def plot_step(ax, step, name, title, scale_acq=True):
    """Plot the GP posterior, the data, and the acquisition function of one iteration."""
    mean, sig, acq = step['mean'], step['sig'], step['acq']
    ax.plot(x, true_func(x), '--', color='C0', label='Ground truth')
    ax.errorbar(step['x_d'], step['y_d'], sigma_e*z95, fmt='ro', capsize=3, label='Data')
    ax.plot(x, mean, color='C1', label='Mean')
    ax.fill_between(x, mean-sig*z95, mean+sig*z95, color='C0', alpha=0.3, label='95% credible interval')
    if scale_acq:
        # EI has a different unit and scale, so it is normalized to the peak height 5 for display
        ax.plot(x, acq/np.max(acq)*5., color='C2', label=f'{name} (scaled)')
    else:
        ax.plot(x, acq, color='C2', label=name)
    ax.axvline(step['x_next'], color='r', ls=':', label='Next point')
    ax.grid()
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_history(history, name, scale_acq=True):
    """Plot all iterations in a grid of panels."""
    n_cols = 4
    n_rows = -(-len(history) // n_cols)
    with plt.rc_context({'font.size': 16}):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5*n_rows), constrained_layout=True)
        for i, (ax, step) in enumerate(zip(axes.flat, history)):
            plot_step(ax, step, name, f'{name} iteration {i+1}', scale_acq)
        for ax in axes.flat[len(history):]:
            ax.set_visible(False)
        axes.flat[0].legend(loc='upper left', fontsize=12)
        plt.show()


def summarize(name, x_d, y_d, hyper=(sigma_0, lscale)):
    """Print the best point estimated by the posterior mean at the measured points."""
    mu_d, _ = gpr(x_d, x_d, y_d, sigma_e, *hyper)
    i_best = np.argmax(mu_d)
    hits = np.flatnonzero(np.abs(x_d - x_opt) < tol_opt)
    first = f'measurement {hits[0]+1}' if len(hits) else 'not reached'
    print(f'{name}: {len(x_d)} measurements, best x = {x_d[i_best]:.2f} (true optimum {x_opt:.2f}), '
          f'posterior mean = {mu_d[i_best]:.2f} (true maximum {true_func(x_opt):.2f}), '
          f'first within |x - x_opt| < {tol_opt}: {first}')



# %% [markdown] id="hWih8jX6KphH"
# ## Take first data before optimization

# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1792389072378, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="fYSKMEpDKphI"
# take first data
rng = np.random.default_rng(seed)
x_d = np.array([x[nx//2]])
y_d = np.array([take_data(x_d[0], rng)])
mean, sig = gpr(x, x_d, y_d, sigma_e, sigma_0, lscale)
first_step = dict(x_d=x_d, y_d=y_d, mean=mean, sig=sig)

# %% [markdown] id="VQt3BIt2KphI"
# ## Plot with EI
# EI is shown in its own unit (not scaled).

# %% colab={"base_uri": "https://localhost:8080/", "height": 582} executionInfo={"elapsed": 2362, "status": "ok", "timestamp": 1792389074739, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="sAItR2jBKphI" outputId="870fe9d8-c529-42b0-db87-64fc8e79ccc3"
# GP with EI
step = dict(first_step, acq=EI(mean, sig, np.max(y_d)))
step['x_next'] = x[np.argmax(step['acq'])]
fig, ax = plt.subplots(figsize=(6,6))
plot_step(ax, step, 'EI', 'GP regression', scale_acq=False)
ax.legend()
plt.show()

# %% [markdown] id="ZmFIPoWOKphJ"
# ## Plot with UCB

# %% colab={"base_uri": "https://localhost:8080/", "height": 582} executionInfo={"elapsed": 447, "status": "ok", "timestamp": 1792389075186, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="I7UPVXrVKphJ" outputId="87c13d9c-f601-4612-f448-46a70ec86ac1"
# GP with UCB
step = dict(first_step, acq=UCB(mean, sig))
step['x_next'] = x[np.argmax(step['acq'])]
fig, ax = plt.subplots(figsize=(6,6))
plot_step(ax, step, 'UCB', 'GP regression', scale_acq=False)
ax.legend()
plt.show()

# %% [markdown] id="BcZTp1rmKphK"
# ## Optimization with EI
# Each panel shows the data used for the regression and the next point (red dotted line) at the maximum of EI.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 4798, "status": "ok", "timestamp": 1792389079987, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="Zy2s-dPpKphK" outputId="6e9d9065-fac9-4e42-b571-1d0afaf487e4"
# Optimization with EI
x_ei, y_ei, history_ei = run_optimization('EI', np.random.default_rng(seed))
plot_history(history_ei, 'EI')
summarize('EI', x_ei, y_ei)

# %% [markdown] id="XsickXpgKphK"
# ## Optimization with UCB

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 2640, "status": "ok", "timestamp": 1792389082630, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="a4JZRylfKphL" outputId="3f3c945a-8434-410b-ccff-f00c941495ab"
# Optimization with UCB
x_ucb, y_ucb, history_ucb = run_optimization('UCB', np.random.default_rng(seed))
plot_history(history_ucb, 'UCB', scale_acq=False)
summarize('UCB', x_ucb, y_ucb)

# %% [markdown]
# ## Optimization with hyper-parameter fitting
# The same EI optimization, but $(\sigma_0, \lambda)$ are fitted by the marginal likelihood at every iteration
# within the loose bounds `bounds_wide`.
# When the first few points happen to lie on a smooth part of the function, a long length scale is fitted.
# Then the uncertainty far from the data is underestimated, and the optimizer can stay at the local maximum.
# Fitting the hyper-parameters from a few points is unreliable without prior knowledge (bounds or priors).

# %%
# Optimization with EI and hyper-parameter fitting
x_eih, y_eih, history_eih = run_optimization('EI', np.random.default_rng(seed), fit_hyper=True)
plot_history(history_eih, 'EI')
for i, step in enumerate(history_eih):
    print(f'iteration {i+1}: sigma_0 = {step["hyper"][0]:.2f}, length scale = {step["hyper"][1]:.2f}')
hyper_final = fit_hyperparameters(x_eih, y_eih, sigma_e, history_eih[-1]['hyper'])
print(f'fitted hyper-parameters: sigma_0 = {hyper_final[0]:.2f}, length scale = {hyper_final[1]:.2f}')
summarize('EI (fitted hyper-parameters)', x_eih, y_eih, hyper_final)

# %% [markdown]
# ## Statistical comparison over many runs
# A single run depends on the random measurement errors.
# The optimization is repeated with `n_runs` different random seeds, and the following quantities are compared:
# - Success rate: the fraction of runs that measured a point within $|x - x_\mathrm{opt}| < 0.3$.
# - First hit: the median index of the first measurement within this tolerance (the first measurement is 1).
# - Points near the local maximum: the mean number of measurements within $|x - x_\mathrm{local}| < 0.5$.
# - Simple regret: $f(x_\mathrm{opt}) - \max_{i \le m} f(x_{\mathcal{D},i})$ after $m$ measurements, averaged over the runs.
#
# Compared settings:
# - EI and UCB with the fixed default hyper-parameters (as in the lecture slides).
# - EI with the posterior-mean incumbent $y^* = \max_i \mu(x_{\mathcal{D},i})$.
# - EI and UCB with the hyper-parameters fitted within the loose bounds ($\lambda \le 5$)
#   and within the narrow bounds ($\lambda \le 1.5$).

# %%
n_runs = 100
tol_local = 0.5
methods = {
    'EI': dict(name='EI'),
    'UCB': dict(name='UCB'),
    'EI, posterior incumbent': dict(name='EI', incumbent='posterior'),
    'EI, fitted (wide)': dict(name='EI', fit_hyper=True, bounds=bounds_wide),
    'UCB, fitted (wide)': dict(name='UCB', fit_hyper=True, bounds=bounds_wide),
    'EI, fitted (narrow)': dict(name='EI', fit_hyper=True, bounds=bounds_narrow),
    'UCB, fitted (narrow)': dict(name='UCB', fit_hyper=True, bounds=bounds_narrow),
}
fig, ax = plt.subplots(figsize=(9, 6))
print(f'{"method":26s} {"success":>8s} {"first hit":>10s} {"near local":>11s}')
for label, options in methods.items():
    runs = [run_optimization(rng=np.random.default_rng(seed+i), **options)[0] for i in range(n_runs)]
    hits = [np.flatnonzero(np.abs(xd - x_opt) < tol_opt) for xd in runs]
    success = np.mean([len(h) > 0 for h in hits])
    first_hit = np.median([h[0]+1 for h in hits if len(h) > 0]) if success > 0. else np.nan
    near_local = np.mean([np.sum(np.abs(xd - x_local) < tol_local) for xd in runs])
    regret = np.mean([true_func(x_opt) - np.maximum.accumulate(true_func(xd)) for xd in runs], axis=0)
    print(f'{label:26s} {success:8.2f} {first_hit:10.1f} {near_local:11.2f}')
    ax.plot(np.arange(1, niter+2), regret, 'o-', label=label)
ax.set_yscale('log')
ax.grid()
ax.legend(fontsize=11)
ax.set_title(f'Mean simple regret over {n_runs} runs')
ax.set_xlabel('Number of measurements')
ax.set_ylabel('Simple regret')
plt.show()
