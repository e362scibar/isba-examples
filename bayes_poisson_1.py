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
# Copyright (C) 2024 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

# %% executionInfo={"elapsed": 16, "status": "ok", "timestamp": 1792388176436, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="dm_FsRzB7OG3"
import numpy as np
import scipy
import matplotlib.pyplot as plt
# %matplotlib inline

# %% executionInfo={"elapsed": 1, "status": "ok", "timestamp": 1792388176442, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="Z6m5Ta0s7OG4"
# Parameters
mu = 2.0
alpha = 1.0
beta = 0.5
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
ax.set_title(f'Poisson distribution $\lambda={mu}$')
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
plt.show()

# %% [markdown] id="AcFJggb17OG6"
# ## Random sampling of Poisson distribution

# %% colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"elapsed": 171, "status": "ok", "timestamp": 1792388177927, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="kUVOWjpg7OG7" outputId="c261935c-95fa-40f4-e90d-365af79dd69b"
# sample n data points
n = 10
dat = poisson.rvs(size=n)
x = np.arange(poisson.ppf(0.01), poisson.ppf(0.99)+1)
hist, bins = np.histogram(dat, bins=len(x), range=(-0.5, len(x)+0.5))
fig, ax = plt.subplots()
ax.plot(x, hist, 'ro', ms=8, label='Data')
ax.vlines(x, 0, hist, colors='r', lw=5, alpha=0.5)
ax.set_title(f'Data')
ax.set_ylabel('Number of samples')
plt.show()

# %% [markdown] id="oj5xGq5V7OG7"
# ## Posterior distribution
# $$
# p(\lambda|\boldsymbol{k}) = \frac{(\beta+n)^{\alpha+\sum_{m=1}^{n}{k_m}} \lambda^{\alpha-1+\sum_{m=1}^{n}{k_m}} e^{-(\beta+n)\lambda}}{\Gamma\left(\alpha+\sum_{m=1}^{b}{k_m}\right)}
# = f\left( \lambda ; \alpha+\sum_{m=1}^{n}{k_m}, \beta+n \right)
# $$

# %% colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"elapsed": 226, "status": "ok", "timestamp": 1792388178175, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="sYHMhe-g7OG7" outputId="68d0c575-cc10-49c6-b5b5-e31c5f8fcdcb"
# Gamma distribution
gamma1 = scipy.stats.gamma(alpha, scale=1./beta)
gamma2 = scipy.stats.gamma(alpha+np.sum(dat), scale=1./(beta+n))
fig, ax = plt.subplots()
x = np.linspace(0., gamma1.ppf(0.99)+1.0, 100)
ax.plot(x, gamma1.pdf(x), 'b-', lw=5, alpha=0.6, label=f'Prior')
ax.plot(x, gamma2.pdf(x), 'r-', lw=5, alpha=0.6, label='Posterior')
ax.grid()
ax.legend()
ax.set_title('Gamma distribution')
ax.set_ylabel('probability density')
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 532} executionInfo={"elapsed": 391, "status": "ok", "timestamp": 1792388178565, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="9_42x9U07OG7" outputId="fa841960-c467-4736-b715-a9ed0d3b2af7"
# Posterior distribution
x = np.arange(0, poisson.ppf(0.99)+1)
pmf = poisson.pmf(x)
pmean = np.empty_like(x)
plow = np.empty_like(x)
phigh = np.empty_like(x)
t = np.linspace(0.001,0.999,999)
for xx in x:
    i = int(xx)
    p = scipy.stats.poisson.pmf(i, gamma2.ppf(t))
    pp = np.sort(p)
    pmean[i] = np.mean(p)
    plow[i] = pmean[i] - pp[158]
    phigh[i] = pp[840] - pmean[i]
fig, ax = plt.subplots()
ax.plot(x, pmf, 'bo', ms=8, label='Ground truth')
ax.vlines(x, 0, pmf, colors='b', lw=5, alpha=0.5)
ax.errorbar(x, pmean, yerr=[plow,phigh], fmt='ro', ms=8, lw=5, alpha=0.5, label='Posterior')
ax.set_title(f'Poisson distribution $\lambda={mu}$')
ax.set_ylabel('Probability')
ax.legend()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 2525, "status": "ok", "timestamp": 1792388181092, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="Kx-N7Qgb7OG8" outputId="1fa52f6c-9c6d-4c5b-d39c-4fd51295a5c8"
# Posteriors
t = np.linspace(0.001,0.999,999)
for xx in x:
    i = int(xx)
    p = scipy.stats.poisson.pmf(i, gamma2.ppf(t))
    fig, ax = plt.subplots(2,1)
    ax[0].hist(p, 50)
    ax[0].set_title(f'k={i}')
    ax[1].plot(p,t)
    plt.show()

# %% executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1792388181129, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -540} id="awnCX4Xl7OG8"
