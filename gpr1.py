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

# %% [markdown] id="_Tu7q3piDjmO"
# # Gaussian process regression
#
# Copyright (C) 2024 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

# %% id="7SXqB1dfDjmP" executionInfo={"status": "ok", "timestamp": 1792388608986, "user_tz": -540, "elapsed": 18, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
import numpy as np
import scipy
import matplotlib.pyplot as plt
# %matplotlib inline

# %% id="6pKVTo1nDjmQ" executionInfo={"status": "ok", "timestamp": 1792388608997, "user_tz": -540, "elapsed": 6, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
# Parameters
plt.rcParams['font.size'] = 16
sigma_0 = 2.0
sigma_e = 0.5
sigma_d = np.sqrt(sigma_0**2 + sigma_e**2)
x_d = 0.0
y_d = 1.0
x = np.linspace(-3.,3.,201)

# %% [markdown] id="BV0Rxt5-DjmR"
# ## Prior distribution
# $$
# E(y)=0
# $$
# $$
# V(y) = \sigma_0^2
# $$

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="pPC74WHeDjmR" executionInfo={"status": "ok", "timestamp": 1792388609237, "user_tz": -540, "elapsed": 224, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="e71aa78c-d869-4e16-826d-51c63ab4ce3c"
# Prior distribution with one point
y = np.ones_like(x)*sigma_0*1.96  # 95 % interval
fig, ax = plt.subplots()
ax.plot(x, np.zeros_like(x), color='b')
ax.fill_between(x, -y, y, color='C0', alpha=0.3, label=r'95% CL')
ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3)
ax.grid()
ax.set_title('Prior distribution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()

# %% [markdown] id="UxZtiIHgDjmS"
# ## Posterior after taking one point
# $$
# E(y) = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_e^2} y_\mathcal{D} \exp{\left(-\frac{1}{2}|x-x_\mathcal{D}|^2\right)}
# $$
# $$V(y) = \sigma_0^2 \left[ 1 - \frac{\sigma_0^2}{\sigma_0^2 + \sigma_e^2} \exp{\left(-\frac{1}{2}|x-x_\mathcal{D}|^2\right)} \right]
# $$

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="uj4C589jDjmS" executionInfo={"status": "ok", "timestamp": 1792388609496, "user_tz": -540, "elapsed": 258, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="355650fa-2c3c-4a9b-fbd9-92e6cc94813c"
# Posterior after taking one data point
mean = sigma_0**2 / sigma_d**2 * y_d * np.exp(-0.5*(x-x_d)**2)
var = sigma_0**2 *1. - sigma_0**4 / sigma_d**2 * np.exp(-(x-x_d)**2)
sigma = np.sqrt(var)
fig, ax = plt.subplots()
ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3, label='Data')
ax.plot(x, mean, color='C1', label='Mean')
ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
ax.grid()
ax.set_title('Posterior distribution with one data point')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()

# %% [markdown] id="5NktE7MHDjmT"
# ## Posterior with 10 data points
# $$
# E(y) = \mathbf{K}_\mathcal{D}^\top \boldsymbol\Sigma_\mathcal{D}^{-1} \boldsymbol{y}_\mathcal{D}
# $$
# $$
# V(y) = \mathbf{K} - \mathbf{K}_\mathcal{D}^\top \boldsymbol\Sigma_\mathcal{D}^{-1} \mathbf{K}_\mathcal{D}
# $$
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
# $$
# \mathbf{K}_\mathcal{D} = \left(
# \begin{matrix}
#   K(x_{\mathcal{D},1},x_1)+\sigma_e^2 & K(x_{\mathcal{D},1},x_2) & \cdots & K(x_{\mathcal{D},1},x_N) \\
#   K(x_{\mathcal{D},2},x_1) & K(x_{\mathcal{D},2},x_2)+\sigma_e^2 & \cdots & K(x_{\mathcal{D},2},x_N) \\
#   \vdots & \vdots & \ddots & \vdots \\
#   K(x_{\mathcal{D},N_\mathcal{D}},x_1) & K(x_{\mathcal{D},N_\mathcal{D}},x_2) & \cdots & K(x_{\mathcal{D},N_\mathcal{D}},x_N)+\sigma_e^2
# \end{matrix}
# \right)
# $$
# $$
# K(x_a,x_b) = \sigma_0^2 \exp{\left(-\frac{1}{2}|x_a - x_b|^2\right)}
# $$

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="YUQotgAVDjmT" executionInfo={"status": "ok", "timestamp": 1792388609743, "user_tz": -540, "elapsed": 246, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="fd67ce71-9269-4222-f684-31f8a6addd45"
# Posterior with 10 points
xmin = 0.
xmax = 10.
n = 10
nx = 201
x = np.linspace(xmin, xmax, nx)
x_d = np.random.uniform(xmin, xmax, n)

def func(x):
    return x * np.sin(x)

y_d = func(x_d) + np.random.randn(n)*sigma_e
dx = np.vstack([x_d]*n) - np.vstack([x_d]*n).T
Sigma = sigma_0**2 * np.exp(-0.5*dx**2)
Sigma += np.eye(n)*sigma_e**2
dx = np.vstack([x]*n) - np.vstack([x_d]*nx).T
kd = sigma_0**2 * np.exp(-0.5*dx**2)
dx = np.vstack([x]*nx) - np.vstack([x]*nx).T
k = sigma_0**2 * np.exp(-0.5*dx**2)
mean = kd.T @ np.linalg.inv(Sigma) @ y_d.T
var = k - kd.T @ np.linalg.inv(Sigma) @ kd
sigma = np.sqrt(var.diagonal())

fig, ax = plt.subplots()
ax.plot(x, func(x), '--', color='C0', label='Ground truth')
ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3, label='Data')
ax.plot(x, mean, color='C1', label='Mean')
ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
ax.legend()
ax.grid()
ax.set_ylim([-8,15])
ax.set_title(f'GP regression from {n} points')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# %% [markdown] id="0nGrk7gPDjmT"
# ## Sampling from posterior

# %% colab={"base_uri": "https://localhost:8080/", "height": 351} id="6xA_OV6iDjmU" executionInfo={"status": "ok", "timestamp": 1792388609931, "user_tz": -540, "elapsed": 187, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="88a03c8e-c902-45b8-d5dc-cb345ecf4402"
# Sampling from posterior
y = np.random.multivariate_normal(mean, var, 10)

fig, ax = plt.subplots(figsize=(6,3))
ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
ax.plot(np.vstack([x]*n).T,y.T)
ax.legend()
ax.grid()
ax.set_title('Sampled functions')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# %% [markdown] id="WeWKSQkCDjmU"
# ## Length scale comparison
# $$
# K(x_a,x_b) = \sigma_0^2 \exp{\left(-\frac{1|x_a - x_b|^2}{2 \lambda^2}\right)}
# $$
# - $\lambda$ is the length scale.

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="UKY7v-IDDjmU" executionInfo={"status": "ok", "timestamp": 1792388610087, "user_tz": -540, "elapsed": 156, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="fe3050b6-0c9c-446b-b18e-ea69fbf10ecb"
# Length scale comparison
lscale = 1.0
dx = np.vstack([x_d]*n) - np.vstack([x_d]*n).T
Sigma = sigma_0**2 * np.exp(-0.5*(dx/lscale)**2)
Sigma += np.eye(n)*sigma_e**2
dx = np.vstack([x]*n) - np.vstack([x_d]*nx).T
kd = sigma_0**2 * np.exp(-0.5*(dx/lscale)**2)
dx = np.vstack([x]*nx) - np.vstack([x]*nx).T
k = sigma_0**2 * np.exp(-0.5*(dx/lscale)**2)
mean = kd.T @ np.linalg.inv(Sigma) @ y_d.T
var = k - kd.T @ np.linalg.inv(Sigma) @ kd
sigma = np.sqrt(var.diagonal())

fig, ax = plt.subplots()
ax.plot(x, func(x), '--', color='C0', label='Ground truth')
ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3, label='Data')
ax.plot(x, mean, color='C1', label='Mean')
ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
ax.legend()
ax.grid()
ax.set_ylim([-8,15])
ax.set_title(f'Length scale {lscale}')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="WwIaOvbuDjmU" executionInfo={"status": "ok", "timestamp": 1792388610246, "user_tz": -540, "elapsed": 158, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="bdcb1236-b89d-4819-dc04-9acaa165ffda"
# Length scale comparison
lscale = 3.0
dx = np.vstack([x_d]*n) - np.vstack([x_d]*n).T
Sigma = sigma_0**2 * np.exp(-0.5*(dx/lscale)**2)
Sigma += np.eye(n)*sigma_e**2
dx = np.vstack([x]*n) - np.vstack([x_d]*nx).T
kd = sigma_0**2 * np.exp(-0.5*(dx/lscale)**2)
dx = np.vstack([x]*nx) - np.vstack([x]*nx).T
k = sigma_0**2 * np.exp(-0.5*(dx/lscale)**2)
mean = kd.T @ np.linalg.inv(Sigma) @ y_d.T
var = k - kd.T @ np.linalg.inv(Sigma) @ kd
sigma = np.sqrt(var.diagonal())

fig, ax = plt.subplots()
ax.plot(x, func(x), '--', color='C0', label='Ground truth')
ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3, label='Data')
ax.plot(x, mean, color='C1', label='Mean')
ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
ax.legend()
ax.grid()
ax.set_ylim([-8,15])
ax.set_title(f'Length scale {lscale}')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="boyV5FuMDjmU" executionInfo={"status": "ok", "timestamp": 1792388610389, "user_tz": -540, "elapsed": 143, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="12249710-28c3-4b38-b3b7-e39326bb1574"
# Length scale comparison
lscale = 0.3
dx = np.vstack([x_d]*n) - np.vstack([x_d]*n).T
Sigma = sigma_0**2 * np.exp(-0.5*(dx/lscale)**2)
Sigma += np.eye(n)*sigma_e**2
dx = np.vstack([x]*n) - np.vstack([x_d]*nx).T
kd = sigma_0**2 * np.exp(-0.5*(dx/lscale)**2)
dx = np.vstack([x]*nx) - np.vstack([x]*nx).T
k = sigma_0**2 * np.exp(-0.5*(dx/lscale)**2)
mean = kd.T @ np.linalg.inv(Sigma) @ y_d.T
var = k - kd.T @ np.linalg.inv(Sigma) @ kd
sigma = np.sqrt(var.diagonal())

fig, ax = plt.subplots()
ax.plot(x, func(x), '--', color='C0', label='Ground truth')
ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3, label='Data')
ax.plot(x, mean, color='C1', label='Mean')
ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
ax.legend()
ax.grid()
ax.set_ylim([-8,15])
ax.set_title(f'Length scale {lscale}')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# %% [markdown] id="qv65tV7qDjmV"
# ## Marginal likelihood
# $$
# \ln{p(\mathcal{D}|\mathrm{X}_\mathcal{D})} =
# -\frac{1}{2} \boldsymbol{y}_\mathcal{D}^\top \boldsymbol\Sigma_\mathcal{D}^{-1} \boldsymbol{y}_\mathcal{D}
# -\frac{1}{2} \ln{|\boldsymbol\Sigma_\mathcal{D}^{-1}|}
# -\frac{N_\mathcal{D}}{2} \ln{2\pi}
# $$

# %% colab={"base_uri": "https://localhost:8080/", "height": 998} id="EXNC2zkmDjmV" executionInfo={"status": "ok", "timestamp": 1792388610697, "user_tz": -540, "elapsed": 307, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="4d8ef760-e01d-4a16-f2aa-28faffa2c911"
# Marginal likelihood
def marginal_likelihood(x, y, sigma0, lscale, err):
    dx = np.vstack([x]*len(x)) - np.vstack([x]*len(x)).T
    Sigma = sigma0**2 * np.exp(-0.5*(dx/lscale)**2)
    Sigma += np.eye(n)*err**2
    return -0.5 * y @ np.linalg.inv(Sigma) @ y.T - 0.5 * np.log(np.linalg.det(Sigma))

s0 = np.linspace(0.1, 10, 100)
lh = np.empty_like(s0)
for i,s in enumerate(s0):
    lh[i] = marginal_likelihood(x_d, y_d, s, 1., sigma_e)
fig, ax = plt.subplots()
ax.plot(s0, lh)
ax.grid()
ax.set_title('Marginal likelihood vs sigma_0')
ax.set_ylabel('log likelihood')
ax.set_xlabel('sigma_0')
plt.show()
print(s0[np.argmax(lh)])

ls = np.linspace(0.1, 10, 100)
lh = np.empty_like(ls)
for i,l in enumerate(ls):
    lh[i] = marginal_likelihood(x_d, y_d, 2.0, l, sigma_e)
fig, ax = plt.subplots()
ax.plot(s0, lh)
ax.grid()
ax.set_title('Marginal likelihood vs length scale')
ax.set_ylabel('log likelihood')
ax.set_xlabel('length scale')
plt.show()
print(ls[np.argmax(lh)])


# %% [markdown] id="Z_P0JeypDjmV"
# ## Maximize marginal likelihood

# %% colab={"base_uri": "https://localhost:8080/"} id="azDUusafDjmV" executionInfo={"status": "ok", "timestamp": 1792388611727, "user_tz": -540, "elapsed": 1028, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="caee4f45-eed5-47a1-b7ed-7cf8eeed0b1b"
# maximize marginal likelihood
def wrap_likelihood(x):
    return -marginal_likelihood(x_d, y_d, x[0], x[1], sigma_e)

x0 = [sigma_0, lscale]
res = scipy.optimize.minimize(wrap_likelihood, x0)
print(res)

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="CGezvZqpDjmV" executionInfo={"status": "ok", "timestamp": 1792388612178, "user_tz": -540, "elapsed": 446, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="20eb2a96-21a0-49f7-a40d-c9747aede51a"
# GP regression with the best hyper-parameters
sig0 = res.x[0]
lscale = res.x[1]
dx = np.vstack([x_d]*n) - np.vstack([x_d]*n).T
Sigma = sig0**2 * np.exp(-0.5*(dx/lscale)**2)
Sigma += np.eye(n)*sigma_e**2
dx = np.vstack([x]*n) - np.vstack([x_d]*nx).T
kd = sig0**2 * np.exp(-0.5*(dx/lscale)**2)
dx = np.vstack([x]*nx) - np.vstack([x]*nx).T
k = sig0**2 * np.exp(-0.5*(dx/lscale)**2)
mean = kd.T @ np.linalg.inv(Sigma) @ y_d.T
var = k - kd.T @ np.linalg.inv(Sigma) @ kd
sigma = np.sqrt(var.diagonal())

fig, ax = plt.subplots()
ax.plot(x, func(x), '--', color='C0', label='Ground truth')
ax.errorbar(x_d, y_d, sigma_e*1.96, fmt='ro', capsize=3, label='Data')
ax.plot(x, mean, color='C1', label='Mean')
ax.fill_between(x, mean-sigma*1.96, mean+sigma*1.96, color='C0', alpha=0.3, label=r'95% CL')
ax.legend()
ax.grid()
ax.set_ylim([-8,15])
ax.set_title(f'Best hyper-parameter')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# %% id="o3mrgzolDjmW" executionInfo={"status": "ok", "timestamp": 1792388612179, "user_tz": -540, "elapsed": 1, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}}
