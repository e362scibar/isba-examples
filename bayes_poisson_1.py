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

# %% [markdown] id="NcM1vcGJ7OG2"
# # Bayesian inference of Poisson distribution
#
# Copyright (C) 2026 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

# %% executionInfo={"elapsed": 16, "status": "ok", "timestamp": 1792388176436, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="dm_FsRzB7OG3"
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import gammaln, lambertw
import matplotlib.pyplot as plt
# %matplotlib inline

# %% executionInfo={"elapsed": 1, "status": "ok", "timestamp": 1792388176442, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="Z6m5Ta0s7OG4"
# Parameters
mu = 2.0  # true lambda of the Poisson distribution
alpha = 1.0  # shape of the Gamma prior
beta = 0.5  # rate of the Gamma prior
n = 10  # number of data points
seed = 2026  # random seed for reproducibility
q_low, q_high = 0.16, 0.84  # quantiles of the 68% credible interval
plt.rcParams['font.size'] = 16

# %% [markdown] id="il1pBK2Z7OG4"
# ## Poisson distribution
# $$
# P(k;\lambda) = \frac{\lambda^{k} e^{-\lambda}}{k!}
# $$

# %% executionInfo={"elapsed": 694, "status": "ok", "timestamp": 1792388177136, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="6gRn4PTR7OG5"
poisson = scipy.stats.poisson(mu)

# %% colab={"base_uri": "https://localhost:8080/", "height": 532} executionInfo={"elapsed": 470, "status": "ok", "timestamp": 1792388177605, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="bOcpswlD7OG5" outputId="023b258c-c48e-4f2f-880c-57f5e14d1cd8"
# Probability distribution
fig, ax = plt.subplots()
x = np.arange(0, poisson.ppf(0.99)+1)
pmf = poisson.pmf(x)
ax.plot(x, pmf, 'bo', ms=8, label='Poisson pmf')
ax.vlines(x, 0, pmf, colors='b', lw=5, alpha=0.5)
ax.set_title(rf'Poisson distribution $\lambda={mu}$')
ax.set_xlabel('$k$')
ax.set_ylabel('Probability')
plt.show()

# %% [markdown] id="BQt-ztyj7OG6"
# ## Gamma distribution
# $$
# f(\lambda;\alpha,\beta) = \frac{\beta^\alpha \lambda^{\alpha-1} e^{-\beta\lambda}}{\Gamma(\alpha)}
# $$
# - Conjugate prior of Poisson distribution

# %% colab={"base_uri": "https://localhost:8080/", "height": 436} executionInfo={"elapsed": 150, "status": "ok", "timestamp": 1792388177756, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="YaY7fDMU7OG6" outputId="3e287312-d64e-4168-abff-69b7534cc8cf"
# Gamma distribution
gamma1 = scipy.stats.gamma(alpha, scale=1./beta)
fig, ax = plt.subplots()
x = np.linspace(0., gamma1.ppf(0.99)+1.0, 100)
ax.plot(x, gamma1.pdf(x), 'r-', lw=5, alpha=0.6, label='gamma pdf')
ax.grid()
ax.set_title(rf'Gamma distribution $\alpha={alpha}$, $\beta={beta}$')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('Probability density')
plt.show()

# %% [markdown] id="AcFJggb17OG6"
# ## Random sampling of Poisson distribution
# A seeded random number generator is used so that the data (and all figures below) are reproducible.

# %% colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"elapsed": 171, "status": "ok", "timestamp": 1792388177927, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="kUVOWjpg7OG7" outputId="c261935c-95fa-40f4-e90d-365af79dd69b"
# sample n data points
rng = np.random.default_rng(seed)
dat = poisson.rvs(size=n, random_state=rng)
x = np.arange(0, max(poisson.ppf(0.99), dat.max())+1)
hist = np.bincount(dat, minlength=len(x))
fig, ax = plt.subplots()
ax.plot(x, hist, 'ro', ms=8, label='Data')
ax.vlines(x, 0, hist, colors='r', lw=5, alpha=0.5)
ax.set_title(f'Data ($n={n}$)')
ax.set_xlabel('$k$')
ax.set_ylabel('Number of samples')
plt.show()

# %% [markdown] id="oj5xGq5V7OG7"
# ## Posterior distribution
# $$
# p(\lambda|\boldsymbol{k}) = \frac{(\beta+n)^{\alpha+\sum_{m=1}^{n}{k_m}} \lambda^{\alpha-1+\sum_{m=1}^{n}{k_m}} e^{-(\beta+n)\lambda}}{\Gamma\left(\alpha+\sum_{m=1}^{n}{k_m}\right)}
# = f\left( \lambda ; \alpha+\sum_{m=1}^{n}{k_m}, \beta+n \right)
# $$

# %% colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"elapsed": 226, "status": "ok", "timestamp": 1792388178175, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="sYHMhe-g7OG7" outputId="68d0c575-cc10-49c6-b5b5-e31c5f8fcdcb"
# Gamma distribution
a_post = alpha + np.sum(dat)  # posterior shape
b_post = beta + n  # posterior rate
gamma1 = scipy.stats.gamma(alpha, scale=1./beta)
gamma2 = scipy.stats.gamma(a_post, scale=1./b_post)
fig, ax = plt.subplots()
x = np.linspace(0., gamma1.ppf(0.99)+1.0, 100)
ax.plot(x, gamma1.pdf(x), 'b-', lw=5, alpha=0.6, label='Prior')
ax.plot(x, gamma2.pdf(x), 'r-', lw=5, alpha=0.6, label='Posterior')
ax.axvline(mu, color='k', ls='--', label=r'True $\lambda$')
ax.grid()
ax.legend()
ax.set_title('Gamma distribution')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('Probability density')
plt.show()

# %% [markdown]
# ## Distribution of the model function (numerical method)
# The posterior of $\lambda$ propagates to an uncertainty of the model function $P(k;\lambda)$ for each $k$.
# Here, $\lambda$ values are generated from the posterior by the inverse-CDF method on an evenly spaced quantile grid
# $t_j \in [0.001, 0.999]$, i.e. $\lambda_j = F^{-1}(t_j)$ where $F$ is the CDF of the posterior.
# This is a deterministic version of random sampling from the posterior.
# For each $k$, the mean and the 68% credible interval (16th–84th percentiles) of $P(k;\lambda_j)$ are evaluated.

# %% colab={"base_uri": "https://localhost:8080/", "height": 532} executionInfo={"elapsed": 391, "status": "ok", "timestamp": 1792388178565, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="9_42x9U07OG7" outputId="fa841960-c467-4736-b715-a9ed0d3b2af7"
# Posterior distribution
x = np.arange(0, poisson.ppf(0.99)+1)
pmf = poisson.pmf(x)
pmean = np.empty_like(x)
plow = np.empty_like(x)
phigh = np.empty_like(x)
t = np.linspace(0.001,0.999,999)
lam_samples = gamma2.ppf(t)
for xx in x:
    i = int(xx)
    p = scipy.stats.poisson.pmf(i, lam_samples)
    pmean[i] = np.mean(p)
    plow[i] = pmean[i] - np.quantile(p, q_low)
    phigh[i] = np.quantile(p, q_high) - pmean[i]
fig, ax = plt.subplots()
ax.plot(x, pmf, 'bo', ms=8, label='Ground truth')
ax.vlines(x, 0, pmf, colors='b', lw=5, alpha=0.5)
ax.errorbar(x, pmean, yerr=[plow,phigh], fmt='ro', ms=8, lw=5, alpha=0.5, label='Posterior')
ax.set_title(rf'Poisson distribution $\lambda={mu}$')
ax.set_xlabel('$k$')
ax.set_ylabel('Probability')
ax.legend()
plt.show()

# %% [markdown]
# ## Distribution of the model function (analytic check)
# In this conjugate case, the distribution of $P(k;\lambda)$ under the posterior $\lambda \sim \mathrm{Gamma}(a, b)$,
# with $a = \alpha+\sum_{m}k_m$ and $b = \beta+n$, can also be derived analytically.
# This is useful to validate the numerical method above.
#
# ### Moments
# Since $P(k;\lambda)^j \propto \lambda^{jk} e^{-j\lambda}$, the $j$-th moment reduces to a Gamma integral:
# $$
# \mathbb{E}\left[P(k;\lambda)^j\right]
# = \frac{b^a}{\Gamma(a)\,(k!)^j}\int_0^\infty \lambda^{a+jk-1} e^{-(b+j)\lambda}\,d\lambda
# = \frac{b^a\,\Gamma(a+jk)}{\Gamma(a)\,(k!)^j\,(b+j)^{a+jk}}
# $$
# - $j=1$: the mean is the **negative binomial distribution** $\mathrm{NB}\left(k; a, \frac{b}{b+1}\right)$ (posterior predictive distribution).
# - $j=2$: the variance is $\mathbb{E}[P^2]-\mathbb{E}[P]^2$.
#
# Note that the variance of the negative binomial distribution, $a(b+1)/b^2$, is the spread of the *next observed count* $k$.
# It is different from the spread of the *probability* $P(k;\lambda)$ discussed here.
#
# ### Quantiles
# $P(k;\lambda)$ is not monotonic in $\lambda$: it has a maximum at $\lambda=k$ (monotonically decreasing for $k=0$).
# Thus, $P(k;\lambda)\le p$ holds when $\lambda$ is on either tail of the peak:
# $$
# \Pr\left[P(k;\lambda)\le p\right] = G(\lambda_-) + 1 - G(\lambda_+),
# \qquad
# \lambda_\mp = -k\,W_{0,\,-1}\!\left(-\frac{(k!\,p)^{1/k}}{k}\right),
# $$
# where $G$ is the CDF of the posterior and $\lambda_\mp$ are the two roots of $\lambda^k e^{-\lambda} = k!\,p$,
# given by the principal ($W_0$) and lower ($W_{-1}$) branches of the Lambert $W$ function.
# The quantiles are obtained by solving this equation for $p$ with a 1D root finder.
# For $k=0$, $P=e^{-\lambda}$ is monotonic, so $p_q = \exp\left(-G^{-1}(1-q)\right)$ in closed form.

# %%
def pmf_moment(k, j, a, b):
    """j-th moment of the Poisson pmf P(k; lambda) for lambda ~ Gamma(a, rate=b)."""
    return np.exp(a*np.log(b) + gammaln(a+j*k) - gammaln(a) - j*gammaln(k+1) - (a+j*k)*np.log(b+j))


def pmf_cdf(p, k, posterior):
    """CDF of P(k; lambda) at p for lambda distributed according to the posterior."""
    if k == 0:
        return posterior.sf(-np.log(p))
    z = -(p*np.exp(gammaln(k+1)))**(1./k) / k
    lam_minus = -k*lambertw(z, 0).real
    lam_plus = -k*lambertw(z, -1).real
    return posterior.cdf(lam_minus) + posterior.sf(lam_plus)


def pmf_quantile(q, k, posterior):
    """q-quantile of P(k; lambda) for lambda distributed according to the posterior."""
    if k == 0:
        return np.exp(-posterior.ppf(1.-q))
    p_max = scipy.stats.poisson.pmf(k, k)  # maximum of P(k; lambda) at lambda = k
    p_min = np.finfo(float).tiny
    return brentq(lambda p: pmf_cdf(p, k, posterior) - q, p_min, p_max*(1.-1e-12))


k_values = x.astype(int)
amean = pmf_moment(k_values, 1, a_post, b_post)
astd = np.sqrt(pmf_moment(k_values, 2, a_post, b_post) - amean**2)
alow = np.array([pmf_quantile(q_low, k, gamma2) for k in k_values])
ahigh = np.array([pmf_quantile(q_high, k, gamma2) for k in k_values])
nbinom_pmf = scipy.stats.nbinom.pmf(k_values, a_post, b_post/(b_post+1.))
nstd = np.array([np.std(scipy.stats.poisson.pmf(k, lam_samples)) for k in k_values])

print(' k   mean(num)  mean(ana)  NB pmf     std(num)   std(ana)   q16(num)   q16(ana)   q84(num)   q84(ana)')
for k in k_values:
    print(f'{k:2d}  {pmean[k]:9.5f}  {amean[k]:9.5f}  {nbinom_pmf[k]:9.5f}  {nstd[k]:9.5f}  {astd[k]:9.5f}  '
          f'{pmean[k]-plow[k]:9.5f}  {alow[k]:9.5f}  {pmean[k]+phigh[k]:9.5f}  {ahigh[k]:9.5f}')

# %%
offset = 0.15  # horizontal shift to separate the two error bars
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, pmf, 'bo', ms=8, label='Ground truth')
ax.vlines(x, 0, pmf, colors='b', lw=5, alpha=0.5)
ax.errorbar(x-offset, pmean, yerr=[plow, phigh], fmt='ro', ms=8, lw=5, alpha=0.5, label='Posterior (numerical)')
ax.errorbar(x+offset, amean, yerr=[amean-alow, ahigh-amean], fmt='gs', ms=8, lw=5, alpha=0.5,
            label='Posterior (analytic)')
ax.set_title(rf'Poisson distribution $\lambda={mu}$')
ax.set_xlabel('$k$')
ax.set_ylabel('Probability')
ax.legend()
plt.show()

# %% [markdown]
# ## Distribution of $P(k;\lambda)$ for each $k$
# Histograms of $P(k;\lambda_j)$ with the analytic mean (solid) and 68% credible interval (dashed).
# The interval is very narrow at $k \approx \lambda$, because $\partial P(k;\lambda)/\partial\lambda = 0$ at $\lambda = k$:
# the fluctuation of $\lambda$ hardly changes $P(k;\lambda)$ there.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 2525, "status": "ok", "timestamp": 1792388181092, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="Kx-N7Qgb7OG8" outputId="1fa52f6c-9c6d-4c5b-d39c-4fd51295a5c8"
# Posteriors
n_cols = 4
n_rows = -(-len(k_values) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows), constrained_layout=True)
for ax, k in zip(axes.flat, k_values):
    p = scipy.stats.poisson.pmf(k, lam_samples)
    ax.hist(p, 50, density=True, color='r', alpha=0.5)
    ax.axvline(amean[k], color='g', lw=2)
    ax.axvline(alow[k], color='g', ls='--', lw=2)
    ax.axvline(ahigh[k], color='g', ls='--', lw=2)
    ax.set_title(f'$k={k}$')
    ax.set_xlabel(r'$P(k;\lambda)$')
for ax in axes.flat[len(k_values):]:
    ax.set_visible(False)
plt.show()
