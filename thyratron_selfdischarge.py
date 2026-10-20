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

# %% [markdown] id="ce1d0c98-ea9a-4613-a970-9a7fce2af460"
# # Example of Bayesian inference for the self-discharge count of a thyratron
#
# Copyright (C) 2024 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

# %% id="dd5a5bcb-9c52-4173-a2db-1f11caf207d9"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.stats import gamma, nbinom
plt.rcParams['font.size'] = 16
# %matplotlib inline

# %% [markdown] id="2615747d"
# # Load data

# %% colab={"base_uri": "https://localhost:8080/"} id="sZO6aEE7NHTx" executionInfo={"status": "ok", "timestamp": 1730741237690, "user_tz": -420, "elapsed": 2613, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="371d367c-3d37-4b06-f5ce-b9d73705e9df"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="85b1316c" executionInfo={"status": "ok", "timestamp": 1730741238592, "user_tz": -420, "elapsed": 905, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="c7a7ac6e-0e0c-442a-9550-f716a42bb3e0"
basedir = '/content/drive/MyDrive/Colab Notebooks/ISBA24'
data = pd.read_csv(f"{basedir}/thyratron_selfdischarge.csv")
display(data)

# %% colab={"base_uri": "https://localhost:8080/", "height": 589} id="e2f3e73b-ca79-4bff-9437-4e174d15c468" executionInfo={"status": "ok", "timestamp": 1730741238593, "user_tz": -420, "elapsed": 22, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="b3a078d4-8498-487e-8870-723675b3982e"
# Plot
fig, ax = plt.subplots(2, 1, figsize=(6,6))
ax[0].plot(data["hrs"], data["cnt"])
ax[0].set_xlabel("Hours")
ax[0].set_ylabel("Counts")
ax[0].set_title("CB08-4 Self-discharge Counts in 2021")
ax[0].grid()
ax[1].plot(data["hrs"], data["diff"])
ax[1].set_xlabel("Hours")
ax[1].set_ylabel("Counts/hour")
ax[1].grid()
plt.tight_layout()
plt.show()

# %% [markdown] id="d4489076-312d-4b20-a27c-4b056f28010b"
# # Bayesian inference of the self-discharge counts per hour
# - Assume Poisson distribution.
#   - ${\rm Poi}(n|\lambda) = \frac{\lambda^n}{n!} e^{-\lambda}$
#   - $E(n) = V(n) = \lambda$
# - The conjugate prior function of the Poisson distribution is the Gamma distribution.
#   - ${\rm Gam}(\lambda|\alpha,\beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} \lambda^{\alpha-1} e^{-\beta\lambda},
#     \qquad \alpha > 0, \qquad \beta > 0$
#   - $E(\lambda) = \frac{\alpha}{\beta}$
#   - $\ln {\rm Gam}(\lambda|\alpha,\beta) = (\alpha - 1) \ln \lambda - \beta \lambda + {\rm const.}$
# - If we observe the data ${\bf N} = \{n_1, n_2, \cdots, n_T\}$, the posterior distribution is calculated to be
# $$ p(\lambda|{\bf N}) \propto p({\bf N}|\lambda) p(\lambda)
# = \left\{ \prod_{t=1}^T {\rm Poi}(n_t|\lambda) \right\} {\rm Gam}(\lambda|\alpha,\beta). $$
# - The logarithm of the distribution is
# $$ \begin{align}
# \ln p(\lambda|{\bf N}) &= \sum_{t=1}^T \left[ n_t \ln \lambda - \ln n_t! - \lambda \right]
# + (\alpha -1)\ln \lambda - \beta\lambda + \alpha \ln \beta - \ln \Gamma(\alpha) + {\rm const.} \\
# &= \left( \sum_{t=1}^T n_t + \alpha - 1 \right) \ln \lambda - (T + \beta)\lambda + {\rm const.}
# \end{align} $$
# - This formula is the same as the Gamma distribution. Therefore,
# $$ p(\lambda|{\bf N}) = {\rm Gam}(\lambda|\hat{\alpha},\hat{\beta}), \qquad
# \hat{\alpha} = \sum_{t=1}^T n_t + \alpha, \qquad \hat{\beta} = T+\beta. $$

# %% [markdown] id="5284bab5-f9bc-493b-b105-0c61b4cb7833"
# - Now, we use the mean of the first 100 hours for the prior mean.
# - We set the initial $\alpha$ to 1.

# %% colab={"base_uri": "https://localhost:8080/", "height": 507} id="6a4cd288-9d91-40ea-99ae-53ebfeef86a4" executionInfo={"status": "ok", "timestamp": 1730741239512, "user_tz": -420, "elapsed": 940, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="9b443f27-1e9a-4392-b07a-cd1c4af97976"
# Prior distribution of lambda
hrs0 = 100
a0 = 1.
b0 = a0 / data["diff"][data["hrs"]<hrs0].mean()
prior = gamma(a0, scale=1./b0)
print(f"alpha = {a0}, beta = {b0}")

x = np.linspace(prior.ppf(0.), prior.ppf(0.99), 1000)
fig, ax = plt.subplots()
ax.plot(x, prior.pdf(x))
ax.grid()
ax.set_xlabel("$\lambda$")
ax.set_title("Prior Distribution")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="678fe2c2-135f-4254-ba04-e36d7b3e4aac" executionInfo={"status": "ok", "timestamp": 1730741239512, "user_tz": -420, "elapsed": 30, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="738c0a79-1af4-47d0-fd3b-747ab5648896"
# Posterior parameters
data["a"] = data["cnt"] + a0
data["b"] = data["hrs"] + b0
display(data)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="a00fe615-2552-4e4b-b2fa-d5b79d0959ce" executionInfo={"status": "ok", "timestamp": 1730741243917, "user_tz": -420, "elapsed": 4432, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="4572a1c1-5c5b-4ff1-a028-1c254ddb598b"
# Posterior distribution at every 100 hours
step = 100
for hrs in range(step, data.index.max(), step):
    param = data.loc[hrs]
    post = gamma(param["a"], scale=1./param["b"])
    fig, ax = plt.subplots()
    ax.plot(x, post.pdf(x))
    ax.grid()
    ax.set_title(f"Posterior after {hrs} hours")
    ax.set_xlabel("$\lambda$")
    plt.show()
    print(f"time = {hrs} hours, alpha = {param['a']}, beta = {param['b']}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 868} id="55088922-22d2-46b1-8d60-7bbf985bc057" executionInfo={"status": "ok", "timestamp": 1730741244471, "user_tz": -420, "elapsed": 574, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="65a59126-739b-4ae5-8705-5170f93c2445"
# Evolution of the expectation of lambda
data["E"] = data["a"] / data["b"]
display(data)
fig, ax = plt.subplots()
ax.plot(data["hrs"], data["E"])
ax.set_xlabel("Hours")
ax.set_ylabel("$E(\lambda)$")
ax.grid()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 868} id="543c3cda-a9f3-47b2-982a-987024e4de22" executionInfo={"status": "ok", "timestamp": 1730741247133, "user_tz": -420, "elapsed": 2664, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="ca197d1b-c6b8-4745-ce73-3590de8ed25b"
# with 10% - 90% interval
low = np.empty(len(data))
high = np.empty(len(data))
for i in range(len(data)):
    post = gamma(data["a"].iloc[i], scale=1./data["b"].iloc[i])
    low[i] = post.ppf(0.1)
    high[i] = post.ppf(0.9)
data["low"] = low
data["high"] = high
display(data)
fig, ax = plt.subplots()
ax.plot(data["hrs"], data["E"], label="$E(\lambda)$")
ax.plot(data["hrs"], data["low"], '--', label="lower bound")
ax.plot(data["hrs"], data["high"], '--', label="upper bound")
ax.set_xlabel("Hours")
ax.set_ylabel("$\lambda$")
ax.grid()
ax.legend()
plt.show()

# %% [markdown] id="f65b28ea-4c7b-4832-bcbb-bf1c78011b91"
# # Predicted distribution of the count
# - $n_*$: predicted counts
# $$ \begin{align}
# p(n_*) &= \int p(n_*|\lambda)p(\lambda)d\lambda \\
# &= \int {\rm Poi}(n_*|\lambda) {\rm Gam}(\lambda|\hat{\alpha},\hat{\beta}) d\lambda \\
# &= \frac{\hat{\beta}^{\hat{\alpha}}}{n_*! \Gamma(\hat{\alpha})}
#    \int \lambda^{n_*+\hat{\alpha}-1} e^{-(\hat{\beta}+1)\lambda} d\lambda \\
# &= \frac{\hat{\beta}^{\hat{\alpha}} \Gamma(n_*+\hat{\alpha})}
#         {n_*! (\hat{\beta}+1)^{n_*+\hat{\alpha}} \Gamma(\hat{\alpha})} \\
# &= \frac{\Gamma(n_*+r)}{n_*! \Gamma(r)} (1-p)^r p^{n_*}
# \end{align} $$
# - Here, we defined $r=\hat{\alpha}$, $p=\frac{1}{\hat{\beta}+1}$.
# - This distribution is called negative binomial distribution.
# - $E(n_*) = \frac{pr}{1-p} = \frac{\hat{\alpha}}{\hat{\beta}}$
# - $E(n_*^2) = \frac{pr(pr+1)}{(1-p)^2} = \frac{\hat{\alpha}(\hat{\alpha}+\hat{\beta}+1)}{\hat{\beta}^2}$
# - $V(n_*) = E(n_*^2) - [E(n_*)]^2 = \frac{pr}{(1-p)^2} = \frac{\hat{\alpha}(\hat{\beta}+1)}{\hat{\beta}^2}$

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="648b8313-1819-418f-aa6e-5963b3e7cfce" executionInfo={"status": "ok", "timestamp": 1730741255586, "user_tz": -420, "elapsed": 8456, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="c9bc272d-c05c-484d-f4e2-cadf5b606c7f"
# Predicted distribution at every 100 hours
step = 100
n = np.arange(0, 6, dtype=int)
for hrs in range(step, data.index.max(), step):
    param = data.loc[hrs]
    pred = nbinom(param["a"], param["b"]/(1+param["b"]))
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    for i in range(2):
        ax[i].vlines(n, 0, pred.pmf(n))
        ax[i].plot(n, pred.pmf(n), 'o')
        ax[i].grid()
        ax[i].set_title(f"Prediction after {hrs} hours")
        ax[i].set_xlabel("$n_*$")
    ax[0].set_ylim([None, 1.0])
    ax[1].set_yscale("log")
    ax[1].set_ylim([None, 10.0])
    plt.show()

# %% [markdown] id="f9532a5e-6131-445e-bd09-a58cc6dacae7"
# # Expectation of predicted cumulative counts

# %% colab={"base_uri": "https://localhost:8080/", "height": 597} id="b0666428-3c90-4ce9-9929-963afe970db3" executionInfo={"status": "ok", "timestamp": 1730741255587, "user_tz": -420, "elapsed": 25, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="2ee09cc8-a789-4f2e-f4bb-dffe8cebb3d2"
data["pred"] = data["hrs"]*data["E"]
data["dcnt"] = data["cnt"]-data["pred"]
display(data)

# %% colab={"base_uri": "https://localhost:8080/", "height": 588} id="9e694852-9aae-437f-90f4-fce667ba8aaf" executionInfo={"status": "ok", "timestamp": 1730741256255, "user_tz": -420, "elapsed": 691, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="d8de5ea6-831f-4b28-ff2c-a0681326229c"
# Plot
fig, ax = plt.subplots(2, 1, figsize=(6,6))
ax[0].plot(data["hrs"], data["cnt"], label="Data")
ax[0].plot(data["hrs"], data["pred"], label="Expectation")
ax[0].grid()
ax[0].legend()
ax[0].set_xlabel("Hours")
ax[0].set_ylabel("Counts")
ax[1].plot(data["hrs"], data["dcnt"])
ax[1].grid()
ax[1].set_xlabel("Hours")
ax[1].set_ylabel("Difference")
plt.tight_layout()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 597} id="6a73ed3a-6046-46e2-841d-29050de06aa5" executionInfo={"status": "ok", "timestamp": 1730741256255, "user_tz": -420, "elapsed": 28, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="9ac28454-5857-4b77-9ed2-9de29ccb69e6"
display(data[(data["hrs"]>100)&(data["dcnt"]>1)])

# %% [markdown] id="d6c76d0b-9ca1-443a-9326-5ee926286679"
# - The difference of the cumulative counts exceeds 1 at 1658 hours.
# - This is about 5 days before the death.

# %% [markdown] id="68b5d939-7bd5-480e-9904-7202e32fde59"
# # Distribution of predicted cumulative counts
# - The distribution of the sum of the probability variable $n_*$ must be calculated.
# - It can be calculated by using the moment-generating function (MGF).
#   - ${\rm MGF}_{X}(t) = E(e^{tX}) = \int e^{tX} p(X) dX, \qquad t\in{\mathbb R}, \qquad X$: probability variable.
#   - MGF is something like a Laplace transform of a probability distribution.
#   - Usually, the distribution of the sum of probability variables can be calculating by taking the convolution of these variables.
#   - The Laplace transform of a convolution is a product, so the sum of probability variables can be transformed to a product.
#   - ${\rm MGF}_{X+Y}(t) = {\rm MGF}_{X}(t) {\rm MGF}_{Y}(t)$
# - The MGF of a negative binomial distribution is
#   ${\rm MGF}_{n_*} = \left( \frac{1-p}{1-pe^t} \right)^r = \left( \frac{\hat{\beta}}{\hat{\beta}+1-e^t} \right)^{\hat{\alpha}}$.
# - The MGF of the sum of the negative binomial distribution is
#   ${\rm MGF}_{T n_*} = \left( \frac{1-p}{1-pe^t} \right)^{Tr} = \left( \frac{\hat{\beta}}{\hat{\beta}+1-e^t} \right)^{T\hat{\alpha}}$.
# - Therefore, the distribution of predicted cumulative counts is also a negative binomial distribution.
#   $r=T\hat{\alpha}$, $p=\frac{1}{\hat{\beta}+1}$

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="e6a3571b-b4e4-4932-b1c4-156a9aedee41" executionInfo={"status": "ok", "timestamp": 1730741268792, "user_tz": -420, "elapsed": 12564, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="2d0993f6-4b21-49fb-ba57-28d4b7fce0db"
# Distribution of the predicted cumulative counts at every 100 hours
step = 100
for hrs in range(step, data.index.max(), step):
    param = data.loc[hrs]
    pred = nbinom(param["a"]*hrs, param["b"]/(1+param["b"]))
    nmin = pred.ppf(0.001)
    nmax = pred.ppf(0.999)
    n = np.arange(nmin, nmax+1, dtype=int)
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    for i in range(2):
        ax[i].vlines(n, 0, pred.pmf(n))
        ax[i].plot(n, pred.pmf(n), 'o', label="Prediction")
        if i == 1:
            ax[1].set_yscale("log")
        ylim = ax[i].get_ylim()
        ax[i].vlines(param["cnt"], ylim[0], ylim[1], "C1", label="Data")
        ax[i].set_ylim(ylim)
        ax[i].grid()
        ax[i].set_title(f"Cumulative counts after {hrs} hours")
        ax[i].set_xlabel("$\sum n_*$")
        ax[i].legend()
    plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="850d6bf6-e817-4e76-998b-bd7b3cc86158" executionInfo={"status": "ok", "timestamp": 1730741271574, "user_tz": -420, "elapsed": 2785, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="b6eb4006-7fc3-416a-f7a3-4ede83915688"
# 10% - 90% interval
low = np.empty(len(data))
high = np.empty(len(data))
for i in range(len(data)):
    pred = nbinom(data["a"].iloc[i]*i, data["b"].iloc[i]/(1+data["b"].iloc[i]))
    low[i] = pred.ppf(0.1)
    high[i] = pred.ppf(0.9)
data["low2"] = low
data["high2"] = high
display(data)
fig, ax = plt.subplots(2, 1, figsize=(6, 8))
ax[0].plot(data["hrs"], data["pred"], label="prediction")
ax[0].plot(data["hrs"], data["low2"], '--', label="lower bound")
ax[0].plot(data["hrs"], data["high2"], '--', label="upper bound")
ax[0].set_xlabel("Hours")
ax[0].set_ylabel("$\sum n_*$")
ax[0].grid()
ax[0].legend()
ax[1].plot(data["hrs"], data["cnt"]-data["pred"], label="data-prediction")
ax[1].plot(data["hrs"], data["low2"]-data["pred"], '--', label="lower bound")
ax[1].plot(data["hrs"], data["high2"]-data["pred"], '--', label="upper bound")
ax[1].set_xlabel("Hours")
ax[1].set_ylabel("$\sum n - \sum n_*$")
ax[1].grid()
ax[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown] id="1fbb2825-83cf-4514-99cd-c599d01d03a1"
# - The data is well within the predicted distribution.
# - We need to compare the data with the past distribution (24 hours before etc.).

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="29cc4b6a-a3b4-456f-8b78-42b2ba18bf1a" executionInfo={"status": "ok", "timestamp": 1730741275477, "user_tz": -420, "elapsed": 3905, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="94a4f658-07a5-45b9-b884-473b65598f01"
# Prediction of the cumulative counts from the data 24 hours before
thre = 0.05
deltat = 24
expect = np.zeros(len(data))
low = np.zeros(len(data))
high = np.zeros(len(data))
for i in range(deltat, len(data)):
    param = data.iloc[i-deltat]
    pred = nbinom(param["a"]*i, param["b"]/(1+param["b"]))
    expect[i] = pred.mean()
    low[i] = pred.ppf(thre)
    high[i] = pred.ppf(1.-thre)
data["pred3"] = expect
data["low3"] = low
data["high3"] = high
display(data)
fig, ax = plt.subplots(2, 1, figsize=(6, 6))
ax[0].plot(data["hrs"], data["pred3"], label="prediction")
ax[0].plot(data["hrs"], data["low3"], '--', label="lower bound")
ax[0].plot(data["hrs"], data["high3"], '--', label="upper bound")
ax[0].set_xlabel("Hours")
ax[0].set_ylabel("$\sum n_*(-\Delta t)$")
ax[0].grid()
ax[0].legend()
ax[1].plot(data["hrs"], data["cnt"]-data["pred3"], label="data-prediction")
ax[1].plot(data["hrs"], data["low3"]-data["pred3"], '--', label="lower bound")
ax[1].plot(data["hrs"], data["high3"]-data["pred3"], '--', label="upper bound")
ax[1].set_xlabel("Hours")
ax[1].set_ylabel("$\sum n - \sum n_*(-\Delta t)$")
ax[1].grid()
ax[1].legend()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 554} id="dffd2b3c" executionInfo={"status": "ok", "timestamp": 1730741276288, "user_tz": -420, "elapsed": 814, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="4163f148-985c-46a1-bc46-a6f54dc5da87"
# Prediction of the cumulative counts
fig, ax = plt.subplots(2, 1, figsize=(6, 6))
ax[0].plot(data["hrs"], data["cnt"], label="data")
ax[0].plot(data["hrs"], data["low3"], '--', label="lower bound")
ax[0].plot(data["hrs"], data["high3"], '--', label="upper bound")
ax[0].set_xlabel("Hours")
ax[0].set_ylabel("$\sum n$")
ax[0].grid()
ax[0].legend()
ax[1].plot(data["hrs"], data["cnt"]-data["pred3"], label="data-prediction")
ax[1].plot(data["hrs"], data["low3"]-data["pred3"], '--', label="lower bound")
ax[1].plot(data["hrs"], data["high3"]-data["pred3"], '--', label="upper bound")
ax[1].set_xlabel("Hours")
ax[1].set_ylabel("$\sum n - \sum n_*(-\Delta t)$")
ax[1].grid()
ax[1].legend()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 617} id="e9571e67-cf7f-4c76-be7f-0b75f3d54a74" executionInfo={"status": "ok", "timestamp": 1730741276288, "user_tz": -420, "elapsed": 22, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="f4dd8141-8657-4268-b7d8-78d5dca2568c"
display(data[(data["hrs"]>100)&(data["cnt"]>data["high3"])])

# %% [markdown] id="72a3ea38-88b4-40b1-ac80-dfc3f737d4f4"
# - We set the upper limit to the 95% limit of the distribution 24 hours before.
# - The difference of the cumulative counts exceeds this limit at 1616 hours.
# - This is about 7 days before the death.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="c461f789-c71e-4ea1-9c14-1ffd9391ab65" executionInfo={"status": "ok", "timestamp": 1730741287221, "user_tz": -420, "elapsed": 10954, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="88183a57-daca-488b-e57d-31c928e703cf"
# Distribution of the predicted cumulative counts from the data 24 hours before
step = 100
for hrs in range(step, data.index.max(), step):
    param = data.loc[hrs-deltat]
    pred = nbinom(param["a"]*hrs, param["b"]/(1+param["b"]))
    nmin = pred.ppf(0.001)
    nmax = pred.ppf(0.999)
    n = np.arange(nmin, nmax+1, dtype=int)
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    for i in range(2):
        ax[i].vlines(n, 0, pred.pmf(n))
        ax[i].plot(n, pred.pmf(n), 'o', label="Prediction")
        if i == 1:
            ax[1].set_yscale("log")
        ylim = ax[i].get_ylim()
        ax[i].vlines(data["cnt"].loc[hrs], ylim[0], ylim[1], "C1", label="Data")
        ax[i].set_ylim(ylim)
        ax[i].grid()
        ax[i].set_title(f"Cumulative counts after {hrs} hours")
        ax[i].set_xlabel("$\sum n_*$")
        ax[i].legend()
    plt.show()

# %% id="581bcca9-e556-4f1e-9b6f-4ff7c8829f3c"
