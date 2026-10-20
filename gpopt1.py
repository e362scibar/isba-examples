# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="JEPVI5dsKphB"
# # Gaussian process optimizer
#
# Copyright (C) 2024 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

# %% id="mN7BxlXiKphD" executionInfo={"status": "ok", "timestamp": 1792389072212, "user_tz": -540, "elapsed": 30, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
import numpy as np
import scipy
import matplotlib.pyplot as plt
# %matplotlib inline

# %% id="ZMRz5rkWKphE" executionInfo={"status": "ok", "timestamp": 1792389072230, "user_tz": -540, "elapsed": 15, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
# Parameters
plt.rcParams['font.size'] = 16
sigma_0 = 2.0
sigma_e = 0.5
xmin = 0.
xmax = 10.
nx = 201
x = np.linspace(xmin, xmax, nx)


# %% [markdown] id="qh0BfjXgKphF"
# ## Function definition
# - True function: $y = x \sin{x}$

# %% id="UEhWUigtKphF" executionInfo={"status": "ok", "timestamp": 1792389072237, "user_tz": -540, "elapsed": 6, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
# True function and data taking function
def true_func(x):
    return x * np.sin(x)

def take_data(x):
    return true_func(x) + np.random.randn()*sigma_e


# %% [markdown] id="CCBW7642KphF"
# ## Gaussian process regression

# %% id="FG5SW7uvKphG" executionInfo={"status": "ok", "timestamp": 1792389072373, "user_tz": -540, "elapsed": 133, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
# Gaussian process regression
def gpr(x, xd, yd, err, sig0):
    dx = np.vstack([xd]*len(xd)) - np.vstack([xd]*len(xd)).T
    Sigma = sig0**2 * np.exp(-0.5*dx**2)
    Sigma += np.eye(len(xd))*err**2
    dx = np.vstack([x]*len(xd)) - np.vstack([xd]*len(x)).T
    kd = sig0**2 * np.exp(-0.5*dx**2)
    dx = np.vstack([x]*len(x)) - np.vstack([x]*len(x)).T
    k = sig0**2 * np.exp(-0.5*dx**2)
    mean = kd.T @ np.linalg.inv(Sigma) @ yd.T
    var = k - kd.T @ np.linalg.inv(Sigma) @ kd
    return mean, var


# %% [markdown] id="s7ZyMaUKKphG"
# ## Expected improvement (EI)
# $$
# \alpha_\mathrm{EI}(x) = \int_{y_\mathrm{max}}^\infty{(y-y_\mathrm{max})\, p(y|y_\mathcal{D},\mathrm{X}_\mathcal{D},x)\, dy}
# = (\mu(x) - y_\mathrm{max}) \Phi(t) + \sigma(x) \phi(t)
# $$
# - $t = \frac{\mu(x) - y_\mathrm{max}}{\sigma(x)}$
# - $\Phi(t)$: cumulative distribution function of a normal distribution
# - $\phi(t)$: probability density function of a normal distribution

# %% id="RGosRp0HKphH" executionInfo={"status": "ok", "timestamp": 1792389072377, "user_tz": -540, "elapsed": 6, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
# Expected improvement
def EI(yd, mu, v):
    ymax = np.max(yd)
    sig = np.sqrt(np.diagonal(v))
    t = (mu - ymax) / sig
    return (mu - ymax) * scipy.stats.norm.cdf(t) + sig * scipy.stats.norm.pdf(t)


# %% [markdown] id="LXvM5nQJKphH"
# ## Upper confidence bound (UCB)
# $$
# \alpha_\mathrm{UCB} = \mu(x) + \lambda\sigma(x)
# $$

# %% id="AFFh0d-_KphH" executionInfo={"status": "ok", "timestamp": 1792389072377, "user_tz": -540, "elapsed": 5, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
# Upper confidence bound
def UCB(mu, v, a=1.96):
    sig = np.sqrt(np.diagonal(v))
    return mu + a*sig


# %% [markdown] id="hWih8jX6KphH"
# ## Take first data before optimization

# %% id="fYSKMEpDKphI" executionInfo={"status": "ok", "timestamp": 1792389072378, "user_tz": -540, "elapsed": 5, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
# take first data
x_d = np.array([x[nx//2]])
y_d = np.array([take_data(x_d[0])])

# %% [markdown] id="VQt3BIt2KphI"
# ## Plot with EI

# %% colab={"base_uri": "https://localhost:8080/", "height": 582} id="sAItR2jBKphI" executionInfo={"status": "ok", "timestamp": 1792389074739, "user_tz": -540, "elapsed": 2362, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="870fe9d8-c529-42b0-db87-64fc8e79ccc3"
# GP with EI
mean, var = gpr(x, x_d, y_d, sigma_e, sigma_0)
sigma = np.sqrt(var.diagonal())

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(x, true_func(x), '--', color='C0', label='Ground truth')
ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3, label='Data')
ax.plot(x, mean, color='C1', label='Mean')
ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
ax.plot(x, EI(y_d,mean,var), color='C2', label='EI')
ax.legend()
ax.grid()
ax.set_ylim([-10,20])
ax.set_title(f'GP regression')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# %% [markdown] id="ZmFIPoWOKphJ"
# ## Plot with UCB

# %% colab={"base_uri": "https://localhost:8080/", "height": 582} id="I7UPVXrVKphJ" executionInfo={"status": "ok", "timestamp": 1792389075186, "user_tz": -540, "elapsed": 447, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="87c13d9c-f601-4612-f448-46a70ec86ac1"
# GP with UCB
mean, var = gpr(x, x_d, y_d, sigma_e, sigma_0)
sigma = np.sqrt(var.diagonal())

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(x, true_func(x), '--', color='C0', label='Ground truth')
ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3, label='Data')
ax.plot(x, mean, color='C1', label='Mean')
ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
ax.plot(x, UCB(mean,var), color='C2', label='UCB')
ax.legend()
ax.grid()
ax.set_ylim([-10,20])
ax.set_title(f'GP regression')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# %% [markdown] id="BcZTp1rmKphK"
# ## Optimization with EI

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Zy2s-dPpKphK" executionInfo={"status": "ok", "timestamp": 1792389079987, "user_tz": -540, "elapsed": 4798, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="6e9d9065-fac9-4e42-b571-1d0afaf487e4"
# Optimization with EI
niter = 10

x_d = np.array([x[nx//2]])
y_d = np.array([take_data(x_d[0])])

for i in range(niter):
    mean, var = gpr(x, x_d, y_d, sigma_e, sigma_0)
    sigma = np.sqrt(var.diagonal())
    ei = EI(y_d,mean,var)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(x, true_func(x), '--', color='C0', label='Ground truth')
    ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3, label='Data')
    ax.plot(x, mean, color='C1', label='Mean')
    ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
    ax.plot(x, ei/np.max(ei)*5., color='C2', label='EI')
    ax.legend()
    ax.grid()
    ax.set_ylim([-10,20])
    ax.set_title(f'GP Optimization {i+1}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    x_d = np.append(x_d, x[np.argmax(ei)])
    y_d = np.append(y_d, take_data(x_d[-1]))

# %% [markdown] id="XsickXpgKphK"
# ## Optimization with UCB

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="a4JZRylfKphL" executionInfo={"status": "ok", "timestamp": 1792389082630, "user_tz": -540, "elapsed": 2640, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="3f3c945a-8434-410b-ccff-f00c941495ab"
# Optimization with UCB
niter = 10

x_d = np.array([x[nx//2]])
y_d = np.array([take_data(x_d[0])])

for i in range(niter):
    mean, var = gpr(x, x_d, y_d, sigma_e, sigma_0)
    sigma = np.sqrt(var.diagonal())
    ucb = UCB(mean,var)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(x, true_func(x), '--', color='C0', label='Ground truth')
    ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3, label='Data')
    ax.plot(x, mean, color='C1', label='Mean')
    ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
    ax.plot(x, ucb, color='C2', label='UCB')
    ax.legend()
    ax.grid()
    ax.set_ylim([-10,20])
    ax.set_title(f'GP Optimization {i+1}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    x_d = np.append(x_d, x[np.argmax(ucb)])
    y_d = np.append(y_d, take_data(x_d[-1]))

# %% id="pLq17mwtKphL" executionInfo={"status": "ok", "timestamp": 1792389082635, "user_tz": -540, "elapsed": 2, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
