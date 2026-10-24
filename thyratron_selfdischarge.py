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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma, nbinom
plt.rcParams['font.size'] = 16
# %matplotlib inline

# %% [markdown] id="2615747d"
# # Load data
# The data file is read from the working directory if it exists (local clone of the repository).
# Otherwise, for example in Google Colaboratory, it is downloaded from the GitHub repository of this lecture.
#
# - `hrs`: operation time in hours since the installation of the tube
# - `cnt`: cumulative self-discharge count
# - `diff`: self-discharge count in the last hour
#
# The tube was replaced at the end of the data, so the last time stamp is used as the failure time below.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2613, "status": "ok", "timestamp": 1730741237690, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="sZO6aEE7NHTx" outputId="371d367c-3d37-4b06-f5ce-b9d73705e9df"
DATA_FILE = "thyratron_selfdischarge.csv"
DATA_URL = f"https://raw.githubusercontent.com/e362scibar/isba-examples/main/{DATA_FILE}"
data_source = Path(DATA_FILE) if Path(DATA_FILE).exists() else DATA_URL
data = pd.read_csv(data_source)
assert (data["hrs"] == data.index).all(), "The analysis assumes one row per hour."
failure_hours = int(data["hrs"].iloc[-1])
print(f"Loaded {len(data)} hourly records from {data_source}; the tube was replaced after {failure_hours} hours.")

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} executionInfo={"elapsed": 905, "status": "ok", "timestamp": 1730741238592, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="85b1316c" outputId="c7a7ac6e-0e0c-442a-9550-f716a42bb3e0"
display(data)

# %% colab={"base_uri": "https://localhost:8080/", "height": 589} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1730741238593, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="e2f3e73b-ca79-4bff-9437-4e174d15c468" outputId="b3a078d4-8498-487e-8870-723675b3982e"
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
# - Note that the first 100 hours are then used twice: for the prior and again in the likelihood.
#   This *empirical Bayes* shortcut is convenient, but strictly the prior should come from independent information,
#   e.g. from other tubes of the same type.

# %% colab={"base_uri": "https://localhost:8080/", "height": 507} executionInfo={"elapsed": 940, "status": "ok", "timestamp": 1730741239512, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="6a4cd288-9d91-40ea-99ae-53ebfeef86a4" outputId="9b443f27-1e9a-4392-b07a-cd1c4af97976"
# Prior distribution of lambda
PRIOR_HOURS = 100  # hours used for the prior mean
a0 = 1.
b0 = a0 / data["diff"][data["hrs"]<PRIOR_HOURS].mean()
prior = gamma(a0, scale=1./b0)
print(f"alpha = {a0}, beta = {b0}")

x = np.linspace(prior.ppf(0.), prior.ppf(0.99), 1000)
fig, ax = plt.subplots()
ax.plot(x, prior.pdf(x))
ax.grid()
ax.set_xlabel(r"$\lambda$")
ax.set_title("Prior Distribution")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1730741239512, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="678fe2c2-135f-4254-ba04-e36d7b3e4aac" outputId="738c0a79-1af4-47d0-fd3b-747ab5648896"
# Posterior parameters
data["a"] = data["cnt"] + a0
data["b"] = data["hrs"] + b0
display(data)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 4432, "status": "ok", "timestamp": 1730741243917, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="a00fe615-2552-4e4b-b2fa-d5b79d0959ce" outputId="4572a1c1-5c5b-4ff1-a028-1c254ddb598b"
# Posterior distribution at selected times
SNAPSHOT_HOURS = [100, 400, 700, 1000, 1300, 1600]


def snapshot_axes(n_panels: int, height: float = 3.5):
    """Return a grid of axes with three panels per row."""
    n_cols = 3
    n_rows = -(-n_panels // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, height*n_rows), constrained_layout=True)
    return fig, axes.flat


fig, axes = snapshot_axes(len(SNAPSHOT_HOURS))
for ax, hrs in zip(axes, SNAPSHOT_HOURS):
    param = data.loc[hrs]
    post = gamma(param["a"], scale=1./param["b"])
    ax.plot(x, post.pdf(x))
    ax.grid()
    ax.set_title(rf"{hrs} h: $\hat\alpha$={param['a']:.0f}, $\hat\beta$={param['b']:.0f}", fontsize=13)
    ax.set_xlabel(r"$\lambda$")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 868} executionInfo={"elapsed": 574, "status": "ok", "timestamp": 1730741244471, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="55088922-22d2-46b1-8d60-7bbf985bc057" outputId="65a59126-739b-4ae5-8705-5170f93c2445"
# Evolution of the expectation of lambda
data["E"] = data["a"] / data["b"]
display(data)
fig, ax = plt.subplots()
ax.plot(data["hrs"], data["E"])
ax.set_xlabel("Hours")
ax.set_ylabel(r"$E(\lambda)$")
ax.grid()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 868} executionInfo={"elapsed": 2664, "status": "ok", "timestamp": 1730741247133, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="543c3cda-a9f3-47b2-982a-987024e4de22" outputId="ca197d1b-c6b8-4745-ce73-3590de8ed25b"
# with 10% - 90% interval
INTERVAL = (0.1, 0.9)  # lower and upper quantiles of the credible interval
data["low"] = gamma.ppf(INTERVAL[0], data["a"], scale=1./data["b"])
data["high"] = gamma.ppf(INTERVAL[1], data["a"], scale=1./data["b"])
display(data)
fig, ax = plt.subplots()
ax.plot(data["hrs"], data["E"], label=r"$E(\lambda)$")
ax.plot(data["hrs"], data["low"], '--', label="lower bound")
ax.plot(data["hrs"], data["high"], '--', label="upper bound")
ax.set_xlabel("Hours")
ax.set_ylabel(r"$\lambda$")
ax.grid()
ax.legend()
plt.show()

# %% [markdown] id="f65b28ea-4c7b-4832-bcbb-bf1c78011b91"
# # Predicted distribution of the count
# - $n_*$: predicted count in the next hour
# $$ \begin{align}
# p(n_*) &= \int p(n_*|\lambda)p(\lambda)d\lambda \\
# &= \int {\rm Poi}(n_*|\lambda) {\rm Gam}(\lambda|\hat{\alpha},\hat{\beta}) d\lambda \\
# &= \frac{\hat{\beta}^{\hat{\alpha}}}{n_*! \Gamma(\hat{\alpha})}
#    \int \lambda^{n_*+\hat{\alpha}-1} e^{-(\hat{\beta}+1)\lambda} d\lambda \\
# &= \frac{\hat{\beta}^{\hat{\alpha}} \Gamma(n_*+\hat{\alpha})}
#         {n_*! (\hat{\beta}+1)^{n_*+\hat{\alpha}} \Gamma(\hat{\alpha})} \\
# &= \binom{n_*+r-1}{n_*} p^r (1-p)^{n_*}
# \end{align} $$
# - Here, we defined $r=\hat{\alpha}$, $p=\frac{\hat{\beta}}{\hat{\beta}+1}$.
# - This distribution is called negative binomial distribution, ${\rm NB}(n_*|r, p)$.
#   With this definition of $p$ (the "success" probability), it is `scipy.stats.nbinom(r, p)`.
# - $E(n_*) = \frac{r(1-p)}{p} = \frac{\hat{\alpha}}{\hat{\beta}}$
# - $V(n_*) = \frac{r(1-p)}{p^2} = \frac{\hat{\alpha}(\hat{\beta}+1)}{\hat{\beta}^2}$

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 8456, "status": "ok", "timestamp": 1730741255586, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="648b8313-1819-418f-aa6e-5963b3e7cfce" outputId="c9bc272d-c05c-484d-f4e2-cadf5b606c7f"
# Predicted distribution at selected times
n = np.arange(0, 6, dtype=int)
fig, axes = plt.subplots(2, len(SNAPSHOT_HOURS), figsize=(18, 6), constrained_layout=True)
for column, hrs in enumerate(SNAPSHOT_HOURS):
    param = data.loc[hrs]
    pred = nbinom(param["a"], param["b"]/(1+param["b"]))
    for row in range(2):
        ax = axes[row, column]
        ax.vlines(n, 0, pred.pmf(n))
        ax.plot(n, pred.pmf(n), 'o')
        ax.grid()
        ax.set_xlabel("$n_*$")
    axes[0, column].set_title(f"After {hrs} hours", fontsize=13)
    axes[0, column].set_ylim([None, 1.0])
    axes[1, column].set_yscale("log")
    axes[1, column].set_ylim([None, 10.0])
plt.show()

# %% [markdown] id="f9532a5e-6131-445e-bd09-a58cc6dacae7"
# # Expectation of predicted cumulative counts

# %% colab={"base_uri": "https://localhost:8080/", "height": 597} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1730741255587, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="b0666428-3c90-4ce9-9929-963afe970db3" outputId="2ee09cc8-a789-4f2e-f4bb-dffe8cebb3d2"
data["pred"] = data["hrs"]*data["E"]
data["dcnt"] = data["cnt"]-data["pred"]
display(data)

# %% colab={"base_uri": "https://localhost:8080/", "height": 588} executionInfo={"elapsed": 691, "status": "ok", "timestamp": 1730741256255, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="9e694852-9aae-437f-90f4-fce667ba8aaf" outputId="d8de5ea6-831f-4b28-ff2c-a0681326229c"
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 597} executionInfo={"elapsed": 28, "status": "ok", "timestamp": 1730741256255, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="6a73ed3a-6046-46e2-841d-29050de06aa5" outputId="9ac28454-5857-4b77-9ed2-9de29ccb69e6"
exceed_hours = data["hrs"][(data["hrs"]>PRIOR_HOURS)&(data["dcnt"]>1)]
display(data.loc[exceed_hours.index])
first_exceed = int(exceed_hours.iloc[0])
print(f"The difference first exceeds 1 at {first_exceed} hours, "
      f"{failure_hours - first_exceed} hours ({(failure_hours - first_exceed)/24:.1f} days) before the replacement.")

# %% [markdown] id="d6c76d0b-9ca1-443a-9326-5ee926286679"
# - The difference of the cumulative counts exceeds 1 about 5 days before the tube was replaced (see the printed
#   output above). This simple criterion, however, has no probabilistic meaning yet.

# %% [markdown] id="68b5d939-7bd5-480e-9904-7202e32fde59"
# # Distribution of predicted cumulative counts
# We want the distribution of the cumulative count $\sum_{t=1}^T n_t$ predicted by the posterior.
#
# **Pitfall.** The predicted counts $n_t$ of different hours are *not* independent under the posterior: they all
# share the same unknown $\lambda$. They are independent only *given* $\lambda$. Therefore, the sum must be computed
# conditionally on $\lambda$ first, and $\lambda$ is integrated out afterwards.
#
# - The moment-generating function (MGF) is useful for sums of **independent** random variables.
#   - ${\rm MGF}_{X}(t) = E(e^{tX}) = \int e^{tX} p(X) dX, \qquad t\in{\mathbb R}, \qquad X$: random variable.
#   - MGF is something like a Laplace transform of a probability distribution.
#   - The distribution of the sum of independent random variables is the convolution of their distributions.
#   - The Laplace transform of a convolution is a product: ${\rm MGF}_{X+Y}(t) = {\rm MGF}_{X}(t) {\rm MGF}_{Y}(t)$.
# - Given $\lambda$, the hourly counts are independent Poisson variables with ${\rm MGF}_{n_t|\lambda}(t) = e^{\lambda(e^t-1)}$, so
# $$ {\rm MGF}_{\sum n_t|\lambda}(t) = \left[e^{\lambda(e^t-1)}\right]^T = e^{T\lambda(e^t-1)}, $$
#   i.e. $\sum_{t=1}^T n_t \,|\, \lambda \sim {\rm Poi}(T\lambda)$.
# - Integrating over the posterior ${\rm Gam}(\lambda|\hat\alpha,\hat\beta)$ gives
# $$ {\rm MGF}_{\sum n_t}(t) = E_\lambda\left[e^{T\lambda(e^t-1)}\right]
#    = \left( \frac{\hat{\beta}}{\hat{\beta}+T-Te^t} \right)^{\hat{\alpha}}, $$
#   which is the MGF of a negative binomial distribution with
# $$ p\left(\sum_{t=1}^T n_t = m\right) = {\rm NB}\left(m \,\middle|\, r=\hat\alpha,\ p=\frac{\hat\beta}{\hat\beta+T}\right)
#    = \binom{m+\hat\alpha-1}{m}\left(\frac{\hat\beta}{\hat\beta+T}\right)^{\hat\alpha}\left(\frac{T}{\hat\beta+T}\right)^{m}. $$
#   For $T=1$ it reduces to the one-hour prediction above.
# - The mean is $T\hat\alpha/\hat\beta$ and the variance is
#   $V = \frac{T\hat\alpha}{\hat\beta}\left(1+\frac{T}{\hat\beta}\right)$: the Poisson variance plus the uncertainty of
#   $\lambda$, which grows with $T^2$ because all hours share the same $\lambda$.
# - If we wrongly took the product of the *marginal* MGFs, $\left[E_\lambda e^{\lambda(e^t-1)}\right]^T$, we would get
#   ${\rm NB}(T\hat\alpha, \hat\beta/(\hat\beta+1))$ with the variance $\frac{T\hat\alpha}{\hat\beta}\left(1+\frac{1}{\hat\beta}\right)$.
#   It treats the uncertainty of $\lambda$ as independent from hour to hour and underestimates the spread when
#   $T$ is comparable to $\hat\beta$. The panels below show both for comparison.

# %%
def cumulative_prediction(a: float, b: float, hours: float):
    """Posterior predictive distribution of the count in `hours` hours for lambda ~ Gamma(a, rate=b)."""
    return nbinom(a, b/(b+hours))


def independent_sum_prediction(a: float, b: float, hours: float):
    """Incorrect prediction that treats the hourly predictive counts as independent (for comparison only)."""
    return nbinom(a*hours, b/(b+1))

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 12564, "status": "ok", "timestamp": 1730741268792, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="e6a3571b-b4e4-4932-b1c4-156a9aedee41" outputId="2d0993f6-4b21-49fb-ba57-28d4b7fce0db"
# Distribution of the predicted cumulative counts at selected times
fig, axes = snapshot_axes(len(SNAPSHOT_HOURS), height=3.8)
for ax, hrs in zip(axes, SNAPSHOT_HOURS):
    param = data.loc[hrs]
    pred = cumulative_prediction(param["a"], param["b"], hrs)
    wrong = independent_sum_prediction(param["a"], param["b"], hrs)
    n = np.arange(pred.ppf(0.001), pred.ppf(0.999)+1, dtype=int)
    ax.vlines(n, 0, pred.pmf(n))
    ax.plot(n, pred.pmf(n), 'o', label="Prediction")
    #ax.plot(n, wrong.pmf(n), 'k:', label="Independent sum\n(incorrect)")
    ax.axvline(param["cnt"], color="C1", label="Data")
    ax.grid()
    ax.set_title(f"Cumulative counts after {hrs} hours", fontsize=13)
    ax.set_xlabel(r"$\sum n_*$")
axes[0].legend(fontsize=10)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 2785, "status": "ok", "timestamp": 1730741271574, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="850d6bf6-e817-4e76-998b-bd7b3cc86158" outputId="b6eb4006-7fc3-416a-f7a3-4ede83915688"
# 10% - 90% interval
pred_all = cumulative_prediction(data["a"], data["b"], data["hrs"])
wrong_all = independent_sum_prediction(data["a"], data["b"], data["hrs"])
data["low2"] = pred_all.ppf(INTERVAL[0])
data["high2"] = pred_all.ppf(INTERVAL[1])
display(data)
fig, ax = plt.subplots(2, 1, figsize=(7, 8))
ax[0].plot(data["hrs"], data["pred"], label="prediction")
ax[0].plot(data["hrs"], data["low2"], '--', label="lower bound")
ax[0].plot(data["hrs"], data["high2"], '--', label="upper bound")
ax[0].set_xlabel("Hours")
ax[0].set_ylabel(r"$\sum n_*$")
ax[0].grid()
ax[0].legend()
ax[1].plot(data["hrs"], data["cnt"]-data["pred"], label="data-prediction")
ax[1].plot(data["hrs"], data["low2"]-data["pred"], '--', label="lower bound")
ax[1].plot(data["hrs"], data["high2"]-data["pred"], '--', label="upper bound")
#ax[1].plot(data["hrs"], wrong_all.ppf(INTERVAL[1])-data["pred"], 'k:', label="upper bound,\nindependent sum (incorrect)")
#ax[1].plot(data["hrs"], wrong_all.ppf(INTERVAL[0])-data["pred"], 'k:')
ax[1].set_xlabel("Hours")
ax[1].set_ylabel(r"$\sum n - \sum n_*$")
ax[1].grid()
ax[1].legend(fontsize=11)
fig.tight_layout()
plt.show()
for hrs in (500, 1000, 1500):
    print(f"{hrs} h: std of the prediction = {pred_all.std()[hrs]:.2f}, "
          f"independent sum (incorrect) = {wrong_all.std()[hrs]:.2f}")

# %% [markdown] id="1fbb2825-83cf-4514-99cd-c599d01d03a1"
# - The data is well within the predicted distribution, even after the self-discharge rate has increased.
#   The posterior always includes the latest data, so it follows the data and cannot reveal an anomaly by itself.
# - We need to compare the data with a prediction made in the past (24 hours before etc.).

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 3905, "status": "ok", "timestamp": 1730741275477, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="29cc4b6a-a3b4-456f-8b78-42b2ba18bf1a" outputId="94a4f658-07a5-45b9-b884-473b65598f01"
# Prediction of the cumulative counts from the data 24 hours before (approach used in 2024)
UPPER_PROB = 0.95  # upper quantile of the predictive distribution
DELTA_T = 24  # hours
data["pred3"] = 0.
data["low3"] = 0.
data["high3"] = 0.
for i in range(DELTA_T, len(data)):
    param = data.iloc[i-DELTA_T]
    pred = cumulative_prediction(param["a"], param["b"], i)
    data.loc[i, "pred3"] = pred.mean()
    data.loc[i, "low3"] = pred.ppf(1.-UPPER_PROB)
    data.loc[i, "high3"] = pred.ppf(UPPER_PROB)
display(data)
fig, ax = plt.subplots(2, 1, figsize=(7, 7))
ax[0].plot(data["hrs"], data["cnt"], label="data")
ax[0].plot(data["hrs"], data["low3"], '--', label="lower bound")
ax[0].plot(data["hrs"], data["high3"], '--', label="upper bound")
ax[0].set_xlabel("Hours")
ax[0].set_ylabel(r"$\sum n$")
ax[0].grid()
ax[0].legend()
ax[1].plot(data["hrs"], data["cnt"]-data["pred3"], label="data-prediction")
ax[1].plot(data["hrs"], data["low3"]-data["pred3"], '--', label="lower bound")
ax[1].plot(data["hrs"], data["high3"]-data["pred3"], '--', label="upper bound")
ax[1].set_xlabel("Hours")
ax[1].set_ylabel(r"$\sum n - \sum n_*(-\Delta t)$")
ax[1].grid()
ax[1].legend()
fig.tight_layout()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 617} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1730741276288, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="e9571e67-cf7f-4c76-be7f-0b75f3d54a74" outputId="f4dd8141-8657-4268-b7d8-78d5dca2568c"
alarms_cumulative = data["hrs"][(data["hrs"]>PRIOR_HOURS)&(data["cnt"]>data["high3"])]
display(data.loc[alarms_cumulative.index].head())
first_alarm_cumulative = int(alarms_cumulative.iloc[0])
print(f"First alarm at {first_alarm_cumulative} hours, {failure_hours - first_alarm_cumulative} hours "
      f"({(failure_hours - first_alarm_cumulative)/24:.1f} days) before the replacement; "
      f"alarms before 1400 hours: {int(np.sum(alarms_cumulative < 1400))}")

# How many extra counts in the last 24 hours does the upper bound tolerate?
for hrs in (500, 1000, 1500):
    expected_recent = DELTA_T * data["E"][hrs-DELTA_T]
    tolerated = data["high3"][hrs] - data["cnt"][hrs-DELTA_T]
    print(f"{hrs} h: the upper bound allows {tolerated:.0f} counts in the last {DELTA_T} hours "
          f"(expected {expected_recent:.1f})")

# %% [markdown] id="72a3ea38-88b4-40b1-ac80-dfc3f737d4f4"
# - We set the upper limit to the 95% limit of the distribution of the cumulative count predicted from the posterior
#   24 hours before. The count exceeds this limit about 6.5 days before the tube was replaced (see the printed output).
#   (With the incorrect independent-sum distribution used in 2024, the upper bound was narrower and the alarm came
#   a few hours earlier, at 1616 hours.)
# - **Caveat.** This predicts the *whole* cumulative count since $t=0$, although the counts up to $t-\Delta t$ have
#   already been observed. The predicted spread therefore contains the uncertainty of all past hours, and the
#   upper bound tolerates many more counts in the last 24 hours than expected (printed above). The detector is very
#   tolerant: it reacts only to a large increase, and the "95%" does **not** mean a false-alarm probability of 5%.
#   The next section builds a detector whose threshold has a clear meaning.

# %% [markdown]
# # Anomaly detection with the predicted count of the last $\Delta t$ hours
# At time $t$, the posterior from $\Delta t$ hours before, ${\rm Gam}(\lambda|\hat\alpha(t-\Delta t), \hat\beta(t-\Delta t))$,
# predicts the count in the *new* interval $(t-\Delta t, t]$, which has not been used for the inference:
# $$ \Delta n(t) = \sum_{t'=t-\Delta t+1}^{t} n_{t'} \sim
#    {\rm NB}\left(r=\hat\alpha(t-\Delta t),\ p=\frac{\hat\beta(t-\Delta t)}{\hat\beta(t-\Delta t)+\Delta t}\right). $$
# The **tail probability** $P\left(\Delta n \ge \Delta n_\mathrm{obs}(t)\right)$ is the probability of observing at least
# the measured count if the tube behaves as before. An alarm is raised when it falls below a threshold $\epsilon$.
#
# **Choosing the threshold.** If the Poisson model were exact, $\epsilon$ would be the false-alarm probability per
# check. Real data deviate from the model: the self-discharges occasionally come in bursts, and the rate slowly
# drifts with aging. Therefore, the false-alarm rate is evaluated on a reference period in which the tube is known
# to be normal (here the first 1400 hours), and the threshold is set with a safety margin below the smallest tail
# probability seen there. In practice, the threshold should be set on the history of other tubes, not on the tube
# under test.

# %%
REFERENCE_END_HOURS = 1400  # end of the period regarded as normal
THRESHOLDS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
SAFETY_FACTOR = 10.  # margin below the smallest tail probability in the reference period

check_hours = np.arange(DELTA_T, len(data))
past = data.iloc[check_hours - DELTA_T]
increment = data["cnt"].values[check_hours] - past["cnt"].values
increment_prediction = cumulative_prediction(past["a"].values, past["b"].values, DELTA_T)
tail_probability = increment_prediction.sf(increment - 1)  # P(X >= observed increment)
in_reference = (check_hours > PRIOR_HOURS) & (check_hours < REFERENCE_END_HOURS)
after_reference = check_hours >= REFERENCE_END_HOURS


def first_alarm(threshold: float) -> int | None:
    """Return the first alarm hour after the reference period, or None."""
    alarm_hours = check_hours[after_reference & (tail_probability < threshold)]
    return int(alarm_hours[0]) if alarm_hours.size else None


print(f"{'threshold':>10s} {'false-alarm hours':>18s} {'first alarm':>12s} {'days before replacement':>24s}")
for threshold in THRESHOLDS:
    false_alarms = int(np.sum(in_reference & (tail_probability < threshold)))
    alarm = first_alarm(threshold)
    days = f"{(failure_hours - alarm)/24:.1f}" if alarm is not None else "-"
    print(f"{threshold:10.0e} {false_alarms:18d} {str(alarm):>12s} {days:>24s}")

min_reference_probability = float(np.min(tail_probability[in_reference]))
selected_threshold = min_reference_probability / SAFETY_FACTOR
selected_alarm = first_alarm(selected_threshold)
print(f"Smallest tail probability in the reference period: {min_reference_probability:.1e} "
      f"at {int(check_hours[in_reference][np.argmin(tail_probability[in_reference])])} hours")
print(f"Selected threshold {selected_threshold:.1e}: first alarm at {selected_alarm} hours, "
      f"{(failure_hours - selected_alarm)/24:.1f} days before the replacement")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.semilogy(check_hours, tail_probability, lw=1, label=rf"$P(\Delta n \geq \Delta n_\mathrm{{obs}})$, $\Delta t$={DELTA_T} h")
ax.axhline(selected_threshold, color="C3", linestyle="--", label=f"threshold {selected_threshold:.0e}")
ax.axvspan(PRIOR_HOURS, REFERENCE_END_HOURS, color="gray", alpha=0.15, label="reference period")
ax.axvline(selected_alarm, color="C3", linestyle=":", label=f"first alarm ({selected_alarm} h)")
ax.set_ylim(1e-12, 2.)
ax.set_xlabel("Hours")
ax.set_ylabel("Tail probability")
ax.grid()
ax.legend(fontsize=11, loc="lower left")
plt.show()

# %% [markdown]
# - With the threshold set below the smallest tail probability of the reference period, the alarm comes about one
#   week before the replacement, similar to the 2024 approach, but now the threshold has a clear meaning and its
#   false-alarm behaviour has been checked.
# - A threshold of 1% or 0.1%, which would be reasonable for an exact Poisson process, gives many false alarms in the
#   normal period (table above). The deviations from the model (bursts, slow drift of the rate) must be taken into
#   account when an anomaly detector is designed.
# - Possible improvements (exercises):
#   1. Use a *forgetting factor*, $\hat\alpha \leftarrow \gamma\hat\alpha + n_t$ and $\hat\beta \leftarrow \gamma\hat\beta + 1$
#      with $\gamma < 1$, so that the posterior follows the slow drift of the rate.
#   2. Vary $\Delta t$ (e.g. 72 or 168 hours) and compare the false alarms and the detection time.
#   3. Model the bursts with an overdispersed distribution (e.g. a negative binomial instead of a Poisson likelihood).
