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

# %% [markdown] id="_Tu7q3piDjmO"
# # Gaussian process regression
#
# Copyright (C) 2026 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.


# %% executionInfo={"elapsed": 18, "status": "ok", "timestamp": 1792388608986, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="7SXqB1dfDjmP"
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
# %matplotlib inline

# %% executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1792388608997, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="6pKVTo1nDjmQ"
# Parameters
plt.rcParams['font.size'] = 16
seed = 2026  # random seed for reproducibility
z95 = scipy.stats.norm.ppf(0.975)  # half width of the 95% interval in units of the standard deviation
sigma_0 = 2.0  # prior standard deviation
sigma_e = 0.5  # measurement error
sigma_d = np.sqrt(sigma_0**2 + sigma_e**2)
x_d = 0.0
y_d = 1.0
x = np.linspace(-3.,3.,201)
ylim = (-8., 15.)  # y range of the regression plots

# %% [markdown]
# ## Helper functions
# - `rbf_kernel` computes the kernel matrix $K(x_a, x_b) = \sigma_0^2 \exp\left(-\frac{|x_a-x_b|^2}{2\lambda^2}\right)$ for all pairs of points.
# - `gp_posterior` computes the posterior mean vector and covariance matrix (step 5 in the lecture).
#   The linear systems with $\boldsymbol\Sigma_\mathcal{D}$ are solved by the Cholesky decomposition
#   $\boldsymbol\Sigma_\mathcal{D} = LL^\top$ instead of the explicit inverse matrix, which is numerically more stable.
# - `plot_gp` plots the posterior mean and the 95% credible interval together with the data.

# %%
def rbf_kernel(xa, xb, sigma0, lscale=1.0):
    """RBF kernel matrix K[i, j] = sigma0^2 exp(-|xa[i] - xb[j]|^2 / (2 lscale^2))."""
    dx = np.subtract.outer(xa, xb)
    return sigma0**2 * np.exp(-0.5*(dx/lscale)**2)


def gp_posterior(x_d, y_d, x, sigma0, lscale, err):
    """Posterior mean vector and covariance matrix of the GP at x, given the data (x_d, y_d)."""
    Sigma_d = rbf_kernel(x_d, x_d, sigma0, lscale) + np.eye(len(x_d))*err**2  # Sigma_D(X_D, X_D)
    k_d = rbf_kernel(x_d, x, sigma0, lscale)  # K_D(X_D, X)
    k = rbf_kernel(x, x, sigma0, lscale)  # K(X, X)
    chol = scipy.linalg.cho_factor(Sigma_d)
    mean = k_d.T @ scipy.linalg.cho_solve(chol, y_d)
    cov = k - k_d.T @ scipy.linalg.cho_solve(chol, k_d)
    return mean, cov


def plot_gp(x, mean, cov, x_d, y_d, err, title, truth=None):
    """Plot the posterior mean and the 95% credible interval of the latent function with the data."""
    sigma = np.sqrt(np.clip(cov.diagonal(), 0., None))  # rounding errors can make tiny negative variances
    fig, ax = plt.subplots()
    if truth is not None:
        ax.plot(x, truth(x), '--', color='C0', label='Ground truth')
    ax.errorbar(x_d, y_d, err*z95, fmt='ro', capsize=3, label='Data')
    ax.plot(x, mean, color='C1', label='Mean')
    ax.fill_between(x, mean-sigma*z95, mean+sigma*z95, color='C0', alpha=0.3, label='95% credible interval')
    ax.legend()
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax


# %% [markdown] id="BV0Rxt5-DjmR"
# ## Prior distribution
# For the latent function value $\phi = f(x)$:
# $$
# E(\phi)=0
# $$
# $$
# V(\phi) = \sigma_0^2
# $$

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} executionInfo={"elapsed": 224, "status": "ok", "timestamp": 1792388609237, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="pPC74WHeDjmR" outputId="e71aa78c-d869-4e16-826d-51c63ab4ce3c"
# Prior distribution with one point
y = np.ones_like(x)*sigma_0*z95  # half width of the 95% interval
fig, ax = plt.subplots()
ax.plot(x, np.zeros_like(x), color='b')
ax.fill_between(x, -y, y, color='C0', alpha=0.3, label='95% credible interval')
ax.errorbar(x_d, y_d, sigma_e*z95, fmt='ro', capsize=3)
ax.grid()
ax.set_title('Prior distribution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()

# %% [markdown] id="UxZtiIHgDjmS"
# ## Posterior after taking one point
# $$
# E(\phi) = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_e^2} y_\mathcal{D} \exp{\left(-\frac{1}{2}|x-x_\mathcal{D}|^2\right)}
# $$
# $$V(\phi) = \sigma_0^2 \left[ 1 - \frac{\sigma_0^2}{\sigma_0^2 + \sigma_e^2} \exp{\left(-|x-x_\mathcal{D}|^2\right)} \right]
# $$
# The result is cross-checked with the general formula `gp_posterior` for many data points (step 5).

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} executionInfo={"elapsed": 258, "status": "ok", "timestamp": 1792388609496, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="uj4C589jDjmS" outputId="355650fa-2c3c-4a9b-fbd9-92e6cc94813c"
# Posterior after taking one data point
mean = sigma_0**2 / sigma_d**2 * y_d * np.exp(-0.5*(x-x_d)**2)
var = sigma_0**2 *1. - sigma_0**4 / sigma_d**2 * np.exp(-(x-x_d)**2)
mean_general, cov_general = gp_posterior(np.array([x_d]), np.array([y_d]), x, sigma_0, 1.0, sigma_e)
assert np.allclose(mean, mean_general) and np.allclose(var, cov_general.diagonal())
plot_gp(x, mean, np.diag(var), x_d, y_d, sigma_e, 'Posterior distribution with one data point')
plt.show()

# %% [markdown] id="5NktE7MHDjmT"
# ## Posterior with 10 data points
# $$
# \boldsymbol\mu_f(\mathrm{X}) = \mathbf{K}_\mathcal{D}^\top \boldsymbol\Sigma_\mathcal{D}^{-1} \boldsymbol{y}_\mathcal{D},
# \qquad
# \boldsymbol{V}_f(\mathrm{X}) = \mathbf{K} - \mathbf{K}_\mathcal{D}^\top \boldsymbol\Sigma_\mathcal{D}^{-1} \mathbf{K}_\mathcal{D}
# $$
# Kernel matrix of the evaluation points $\mathrm{X} = (x_1, \cdots, x_N)$ ($N \times N$):
# $$
# \mathbf{K} = \left(
# \begin{matrix}
#   K(x_1,x_1) & K(x_1,x_2) & \cdots & K(x_1,x_N) \\
#   K(x_2,x_1) & K(x_2,x_2) & \cdots & K(x_2,x_N) \\
#   \vdots & \vdots & \ddots & \vdots \\
#   K(x_N,x_1) & K(x_N,x_2) & \cdots & K(x_N,x_N)
# \end{matrix}
# \right)
# $$
# Cross-covariance between the data points $\mathrm{X}_\mathcal{D}$ and the evaluation points $\mathrm{X}$ ($N_\mathcal{D} \times N$):
# $$
# \mathbf{K}_\mathcal{D} = \left(
# \begin{matrix}
#   K(x_{\mathcal{D},1},x_1) & K(x_{\mathcal{D},1},x_2) & \cdots & K(x_{\mathcal{D},1},x_N) \\
#   K(x_{\mathcal{D},2},x_1) & K(x_{\mathcal{D},2},x_2) & \cdots & K(x_{\mathcal{D},2},x_N) \\
#   \vdots & \vdots & \ddots & \vdots \\
#   K(x_{\mathcal{D},N_\mathcal{D}},x_1) & K(x_{\mathcal{D},N_\mathcal{D}},x_2) & \cdots & K(x_{\mathcal{D},N_\mathcal{D}},x_N)
# \end{matrix}
# \right)
# $$
# Covariance matrix of the data points including the measurement errors ($N_\mathcal{D} \times N_\mathcal{D}$):
# $$
# \boldsymbol\Sigma_\mathcal{D} = \left(
# \begin{matrix}
#   K(x_{\mathcal{D},1},x_{\mathcal{D},1})+\sigma_e^2 & K(x_{\mathcal{D},1},x_{\mathcal{D},2}) & \cdots & K(x_{\mathcal{D},1},x_{\mathcal{D},N_\mathcal{D}}) \\
#   K(x_{\mathcal{D},2},x_{\mathcal{D},1}) & K(x_{\mathcal{D},2},x_{\mathcal{D},2})+\sigma_e^2 & \cdots & K(x_{\mathcal{D},2},x_{\mathcal{D},N_\mathcal{D}}) \\
#   \vdots & \vdots & \ddots & \vdots \\
#   K(x_{\mathcal{D},N_\mathcal{D}},x_{\mathcal{D},1}) & K(x_{\mathcal{D},N_\mathcal{D}},x_{\mathcal{D},2}) & \cdots & K(x_{\mathcal{D},N_\mathcal{D}},x_{\mathcal{D},N_\mathcal{D}})+\sigma_e^2
# \end{matrix}
# \right)
# $$
# $$
# K(x_a,x_b) = \sigma_0^2 \exp{\left(-\frac{1}{2}|x_a - x_b|^2\right)}
# $$
# - The measurement errors are independent, so $\sigma_e^2$ is added only to the diagonal elements of $\boldsymbol\Sigma_\mathcal{D}$.
# - The shaded band is the 95% credible interval of the latent function $f(x)$.
#   The interval for a new noisy observation $y$ is wider: its variance is $V_f(x)+\sigma_e^2$.

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} executionInfo={"elapsed": 246, "status": "ok", "timestamp": 1792388609743, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="YUQotgAVDjmT" outputId="fd67ce71-9269-4222-f684-31f8a6addd45"
# Posterior with 10 points
xmin = 0.
xmax = 10.
n = 10
nx = 201
rng = np.random.default_rng(seed)
x = np.linspace(xmin, xmax, nx)
x_d = rng.uniform(xmin, xmax, n)

def func(x):
    return x * np.sin(x)

y_d = func(x_d) + rng.normal(0., sigma_e, n)
mean, cov = gp_posterior(x_d, y_d, x, sigma_0, 1.0, sigma_e)
ax = plot_gp(x, mean, cov, x_d, y_d, sigma_e, f'GP regression from {n} points', truth=func)
ax.set_ylim(ylim)
plt.show()


# %% [markdown] id="0nGrk7gPDjmT"
# ## Sampling from posterior
# Functions are sampled from the posterior multivariate normal distribution $\mathcal{N}(\boldsymbol\mu_f, \boldsymbol{V}_f)$.

# %% colab={"base_uri": "https://localhost:8080/", "height": 351} executionInfo={"elapsed": 187, "status": "ok", "timestamp": 1792388609931, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="6xA_OV6iDjmU" outputId="88a03c8e-c902-45b8-d5dc-cb345ecf4402"
# Sampling from posterior
n_samples = 10
y = rng.multivariate_normal(mean, cov, n_samples)
sigma = np.sqrt(np.clip(cov.diagonal(), 0., None))

fig, ax = plt.subplots(figsize=(6,3))
ax.fill_between(x, mean-sigma*z95, mean+sigma*z95, color='C0', alpha=0.3, label='95% credible interval')
ax.plot(x, y.T)
ax.legend()
ax.grid()
ax.set_title('Sampled functions')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# %% [markdown] id="WeWKSQkCDjmU"
# ## Length scale comparison
# $$
# K(x_a,x_b) = \sigma_0^2 \exp{\left(-\frac{|x_a - x_b|^2}{2 \lambda^2}\right)}
# $$
# - $\lambda$ is the length scale.

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} executionInfo={"elapsed": 156, "status": "ok", "timestamp": 1792388610087, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="UKY7v-IDDjmU" outputId="fe3050b6-0c9c-446b-b18e-ea69fbf10ecb"
# Length scale comparison
lscale = 1.0
mean, cov = gp_posterior(x_d, y_d, x, sigma_0, lscale, sigma_e)
ax = plot_gp(x, mean, cov, x_d, y_d, sigma_e, f'Length scale {lscale}', truth=func)
ax.set_ylim(ylim)
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 490} executionInfo={"elapsed": 158, "status": "ok", "timestamp": 1792388610246, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="WwIaOvbuDjmU" outputId="bdcb1236-b89d-4819-dc04-9acaa165ffda"
# Length scale comparison
lscale = 3.0
mean, cov = gp_posterior(x_d, y_d, x, sigma_0, lscale, sigma_e)
ax = plot_gp(x, mean, cov, x_d, y_d, sigma_e, f'Length scale {lscale}', truth=func)
ax.set_ylim(ylim)
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 490} executionInfo={"elapsed": 143, "status": "ok", "timestamp": 1792388610389, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="boyV5FuMDjmU" outputId="12249710-28c3-4b38-b3b7-e39326bb1574"
# Length scale comparison
lscale = 0.3
mean, cov = gp_posterior(x_d, y_d, x, sigma_0, lscale, sigma_e)
ax = plot_gp(x, mean, cov, x_d, y_d, sigma_e, f'Length scale {lscale}', truth=func)
ax.set_ylim(ylim)
plt.show()


# %% [markdown] id="qv65tV7qDjmV"
# ## Marginal likelihood
# $$
# \ln{p(\mathcal{D}|\mathrm{X}_\mathcal{D})} =
# -\frac{1}{2} \boldsymbol{y}_\mathcal{D}^\top \boldsymbol\Sigma_\mathcal{D}^{-1} \boldsymbol{y}_\mathcal{D}
# -\frac{1}{2} \ln{|\boldsymbol\Sigma_\mathcal{D}|}
# -\frac{N_\mathcal{D}}{2} \ln{2\pi}
# $$
# With the Cholesky decomposition $\boldsymbol\Sigma_\mathcal{D} = LL^\top$, the log determinant is computed stably as
# $\ln|\boldsymbol\Sigma_\mathcal{D}| = 2\sum_i \ln L_{ii}$.
# Here, $\sigma_e$ is fixed to the known measurement error, and $\sigma_0$ and $\lambda$ are scanned one by one.

# %% colab={"base_uri": "https://localhost:8080/", "height": 998} executionInfo={"elapsed": 307, "status": "ok", "timestamp": 1792388610697, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="EXNC2zkmDjmV" outputId="4d8ef760-e01d-4a16-f2aa-28faffa2c911"
# Marginal likelihood
def log_marginal_likelihood(x, y, sigma0, lscale, err):
    """Log marginal likelihood ln p(D | X_D) of the GP with the RBF kernel."""
    Sigma_d = rbf_kernel(x, x, sigma0, lscale) + np.eye(len(x))*err**2
    chol = scipy.linalg.cho_factor(Sigma_d)
    log_det = 2. * np.sum(np.log(np.diag(chol[0])))
    return -0.5 * y @ scipy.linalg.cho_solve(chol, y) - 0.5 * log_det - 0.5 * len(x) * np.log(2.*np.pi)

lscale_scan = 1.0
s0 = np.linspace(0.1, 10, 100)
lh = np.array([log_marginal_likelihood(x_d, y_d, s, lscale_scan, sigma_e) for s in s0])
fig, ax = plt.subplots()
ax.plot(s0, lh)
ax.grid()
ax.set_title(f'Marginal likelihood vs $\\sigma_0$ ($\\lambda={lscale_scan}$)')
ax.set_ylabel('log marginal likelihood')
ax.set_xlabel(r'$\sigma_0$')
plt.show()
print(f'best sigma_0 = {s0[np.argmax(lh)]:.2f} (length scale = {lscale_scan})')

ls = np.linspace(0.1, 10, 100)
lh = np.array([log_marginal_likelihood(x_d, y_d, sigma_0, l, sigma_e) for l in ls])
fig, ax = plt.subplots()
ax.plot(ls, lh)
ax.grid()
ax.set_title(f'Marginal likelihood vs $\\lambda$ ($\\sigma_0={sigma_0}$)')
ax.set_ylabel('log marginal likelihood')
ax.set_xlabel(r'length scale $\lambda$')
plt.show()
print(f'best length scale = {ls[np.argmax(lh)]:.2f} (sigma_0 = {sigma_0})')


# %% [markdown] id="Z_P0JeypDjmV"
# ## Maximize marginal likelihood
# Both $\sigma_0$ and $\lambda$ are optimized simultaneously.
# The optimizer works on $\theta = (\ln\sigma_0, \ln\lambda)$, so that the hyper-parameters stay positive without explicit bounds.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1028, "status": "ok", "timestamp": 1792388611727, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="azDUusafDjmV" outputId="caee4f45-eed5-47a1-b7ed-7cf8eeed0b1b"
# maximize marginal likelihood
def neg_log_marginal_likelihood(theta):
    """Negative log marginal likelihood as a function of theta = (ln sigma_0, ln lscale)."""
    sigma0, lscale = np.exp(theta)
    return -log_marginal_likelihood(x_d, y_d, sigma0, lscale, sigma_e)

theta0 = np.log([sigma_0, 1.0])  # initial guess: sigma_0 and length scale 1.0
res = scipy.optimize.minimize(neg_log_marginal_likelihood, theta0)
sig0_best, lscale_best = np.exp(res.x)
print(res.message)
print(f'sigma_0 = {sig0_best:.3f}, length scale = {lscale_best:.3f}, log marginal likelihood = {-res.fun:.3f}')

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} executionInfo={"elapsed": 446, "status": "ok", "timestamp": 1792388612178, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="CGezvZqpDjmV" outputId="20eb2a96-21a0-49f7-a40d-c9747aede51a"
# GP regression with the best hyper-parameters
mean, cov = gp_posterior(x_d, y_d, x, sig0_best, lscale_best, sigma_e)
ax = plot_gp(x, mean, cov, x_d, y_d, sigma_e, 'Best hyper-parameters', truth=func)
ax.set_ylim(ylim)
plt.show()
