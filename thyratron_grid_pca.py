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

# %% [markdown] id="RMnSyHmaPv11"
# # Anomaly detection of a thyratron grid waveform
#
# Copyright (C) 2026 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.


# %% id="JnZHxKS-sYZv"
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['font.size'] = 16
# %matplotlib inline

# %% [markdown] id="9XbraQFJPv12"
# ## Get the data
# The waveform files are expected in the directory `data/`. If it does not exist, the archive `wfmdata.tar.gz` is
# extracted; if the archive does not exist either (e.g. in Google Colaboratory), it is downloaded from the GitHub
# repository of this lecture.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 32005, "status": "ok", "timestamp": 1730741803629, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="EdlqgXAyUOXm" outputId="b4b75e10-ac97-4690-b2f2-27401cd2c764"
DATA_DIR = Path("data")
ARCHIVE = Path("wfmdata.tar.gz")
ARCHIVE_URL = "https://raw.githubusercontent.com/e362scibar/isba-examples/main/wfmdata.tar.gz"

# %% id="yNk3x8RvgcL3"
if not DATA_DIR.is_dir():
    if not ARCHIVE.is_file():
        print(f"Downloading {ARCHIVE_URL}")
        urllib.request.urlretrieve(ARCHIVE_URL, ARCHIVE)
    print(f"Extracting {ARCHIVE}")
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(".", filter="data")

# %% [markdown] id="rO-sisYhfff-"
# # Classification of input data
# Each file `sn<serial>_<hours>h_g1.npz` contains about 100 grid waveforms (160 samples each) of one tube with the
# serial number `<serial>` after `<hours>` hours of operation. Two tubes also have a second file `_g2.npz`, which is
# not used here.
#
# The tubes are classified into three classes:
#
# | Class | Operating hours | Meaning |
# |---|---|---|
# | New | < 5000 h | new tubes |
# | Aged | 5000–20000 h | tubes in operation |
# | Failed | ≥ 20000 h | tubes that actually failed |
#
# In this dataset, the failed tubes are exactly those with at least 20000 hours, so the class is assigned from the
# operating hours in the file name. In general, the labels must come from the maintenance records.

# %% id="HSsP27kCcCie"
NEW_LIMIT_HOURS = 5000
FAILED_LIMIT_HOURS = 20000
CLASS_NAMES = ["new", "aged", "failed"]
NEW, AGED, FAILED = range(3)
FILE_PATTERN = re.compile(r"sn(\d+)_(\d+)h_g1\.npz")


def tube_class(hours: int) -> int:
    """Return the class index of a tube from its operating hours."""
    if hours < NEW_LIMIT_HOURS:
        return NEW
    if hours < FAILED_LIMIT_HOURS:
        return AGED
    return FAILED

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 19551, "status": "ok", "timestamp": 1730741823564, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="cJ8DKcfIsYRT" outputId="7a1d4351-e0ad-423e-8061-77ae1a5d9dc2"
thyIdList = [[], [], []]  # (serial, hours) per class: new, aged, failed
dfList = [[], [], []]  # waveform arrays per class, shape (number of waveforms, samples)
for path in sorted(DATA_DIR.glob("sn*_g1.npz")):
    serial, hours = map(int, FILE_PATTERN.fullmatch(path.name).groups())
    with np.load(path) as npz:
        waveforms = npz["wfm"]
    class_index = tube_class(hours)
    thyIdList[class_index].append((serial, hours))
    dfList[class_index].append(waveforms)

for class_index, name in enumerate(CLASS_NAMES):
    print(f"{name}: {len(thyIdList[class_index])} tubes")

# S/N 2176 (18100 h) seems to be in good condition (see the regression results below)

# %% [markdown] id="lMkM7bU6vc3w"
# ## Training data
# The PCA is fitted to the waveforms of the new and failed tubes. The aged tubes are kept for testing.

# %% id="6wY_-CSHvG7J"
tdf = np.vstack([waveforms for class_index in (NEW, FAILED) for waveforms in dfList[class_index]])
print(f"Training waveforms: {tdf.shape}")

# %% [markdown] id="1g7IQqORPv14"
# # PCA
# The PCA subtracts the mean waveform and decomposes the centered data matrix by the singular value decomposition
# $M - \bar M = U\Sigma V^\top$. The rows of $V^\top$ (`pca.components_`) are the waveform shapes of the principal
# components, and the principal-component scores of the waveforms are the rows of $U\Sigma$ (`pca.transform`).
# The contribution ratio of the $i$-th component is $\sigma_i^2/\sum_j\sigma_j^2$.

# %% colab={"base_uri": "https://localhost:8080/", "height": 80} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1730741823565, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="NWtgW3iHY5aa" outputId="4aab21de-6f21-4d3d-8682-3c76d14c2de4"
N_COMPONENTS = 6
pca = PCA(n_components=N_COMPONENTS)
pca.fit(tdf)

# %% [markdown] id="Rq_92-f0ydE_"
# ## Cumulative contribution ratio

# %% colab={"base_uri": "https://localhost:8080/", "height": 461} executionInfo={"elapsed": 498, "status": "ok", "timestamp": 1730741824050, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="vl93n_TaF3jk" outputId="227681ef-4edd-4d42-f247-7e0c60616585"
cumulative_ratio = np.insert(np.cumsum(pca.explained_variance_ratio_), 0, 0.)
fig, ax = plt.subplots()
ax.plot(cumulative_ratio, 'o-')
ax.set_ylim(0., 1.05)
ax.grid()
ax.set_xlabel('Number of principal components')
ax.set_ylabel('Cumulative contribution ratio')
plt.show()
print("Cumulative contribution ratio:", np.round(cumulative_ratio[1:], 3))

# %% [markdown] id="J_u2ZrcYPv15"
# ## Plot the PCA outputs of the first 10 tubes of each class

# %% colab={"base_uri": "https://localhost:8080/", "height": 976} executionInfo={"elapsed": 1581, "status": "ok", "timestamp": 1730741825615, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="-zsoYq3KPv15" outputId="e896bb6c-0143-45f3-9aed-93a35c21b029"
N_PLOT_TUBES = 10
MARKERS = ['+', 'x', '.']
fig, ax = plt.subplots(figsize=[8, 8])
for class_index, waveform_list in enumerate(dfList):
    for (serial, hours), waveforms in zip(thyIdList[class_index][:N_PLOT_TUBES], waveform_list):
        scores = pca.transform(waveforms)
        ax.scatter(scores[:, 0], scores[:, 1], marker=MARKERS[class_index], label=f'SN: {serial} ({hours}hr)')
ax.set_xlabel('PCA-1')
ax.set_ylabel('PCA-2')
ax.grid()
ax.legend(loc='center left', bbox_to_anchor=(1., 0.5), fontsize=9)
plt.show()

# %% [markdown] id="_GRfPAjKyg_Z"
# ## Plot the PCA outputs of the training data (new and failed tubes)

# %% colab={"base_uri": "https://localhost:8080/", "height": 714} executionInfo={"elapsed": 1283, "status": "ok", "timestamp": 1730741826881, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="FXc_h8SBwD1e" outputId="0c7578bd-ed23-4f12-8215-20cba1876b3b"
fig, ax = plt.subplots(figsize=[8, 8])
for class_index in (NEW, FAILED):
    for (serial, hours), waveforms in zip(thyIdList[class_index][:N_PLOT_TUBES], dfList[class_index]):
        scores = pca.transform(waveforms)
        ax.scatter(scores[:, 0], scores[:, 1], marker='+' if class_index == NEW else '.', label=f'SN: {serial} ({hours}hr)')
ax.set_xlabel('PCA-1')
ax.set_ylabel('PCA-2')
ax.grid()
ax.legend(loc='center left', bbox_to_anchor=(1., 0.5), fontsize=9)
plt.show()

# %% [markdown] id="QbkrVgMXPv16"
# ## Waveform of each component

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 834, "status": "ok", "timestamp": 1730741827684, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="u4wrzPWvPv16" outputId="e095cbed-2aa3-43f0-e8cc-2b760237c3f8"
fig, axes = plt.subplots(2, 3, figsize=(16, 7), constrained_layout=True)
for component_index, ax in enumerate(axes.flat):
    ax.plot(pca.components_[component_index, :])
    ax.grid()
    ax.set_title(f'Component {component_index+1}')
    ax.set_xlabel('Sample index')
plt.show()

# %% [markdown] id="oScMUc6UPv16"
# ## Linear regression
# The principal-component scores $x_i$ ($i = 1, \dots, N$) of each waveform are fed into a linear regression
# $$ y = \sum_{i=1}^{N} w_i x_i + b, $$
# trained with $y = 0$ for the new tubes and $y = 1$ for the failed tubes. The aged tubes are used as a test dataset.
# (A linear regression on 0/1 labels is a *linear probability model*; a logistic regression is the more common choice
# for classification.)

# %% id="UjR6Vw26Pv16"
from sklearn.linear_model import LinearRegression

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 429, "status": "ok", "timestamp": 1730741828106, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="F9HKclgQPv16" outputId="cccd764f-e630-4879-bb94-1850106f7946"
# Select new and failed tubes for training and aged tubes for testing
xtrain = np.vstack([pca.transform(waveforms) for class_index in (NEW, FAILED) for waveforms in dfList[class_index]])
ytrain = np.hstack([
    np.full(len(waveforms), float(class_index == FAILED)) for class_index in (NEW, FAILED) for waveforms in dfList[class_index]
])
xtest = np.vstack([pca.transform(waveforms) for waveforms in dfList[AGED]])
print(xtrain.shape, ytrain.shape)
print(xtest.shape)

# %% id="StTQ2zeoPv17"
# Fit
SCORE_RANGE = (-0.5, 1.5)
N_BINS = 100


def plot_score_histograms(model, title: str) -> None:
    """Histogram the regression output for the training (new, failed) and test (aged) waveforms."""
    fig, ax = plt.subplots()
    ax.hist(model.predict(xtrain[ytrain == 0]), N_BINS, range=SCORE_RANGE, alpha=0.8, label='train new')
    ax.hist(model.predict(xtrain[ytrain == 1]), N_BINS, range=SCORE_RANGE, alpha=0.8, label='train failed')
    ax.hist(model.predict(xtest), N_BINS, range=SCORE_RANGE, alpha=0.8, label='test aged')
    ax.grid()
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Regression output')
    plt.show()


linear_model = LinearRegression().fit(xtrain, ytrain)
plot_score_histograms(linear_model, 'Linear regression')

# %% [markdown] id="Aos251ssPv17"
# ## Support vector machine (SVM) regression
# `SVR` (support vector regression) with the RBF kernel
# $K(\boldsymbol{x}, \boldsymbol{x}') = \exp(-\gamma\|\boldsymbol{x}-\boldsymbol{x}'\|^2)$ fits a function that may deviate
# from the target by at most $\epsilon$ without penalty, while keeping the function as flat as possible.
# Errors larger than $\epsilon$ are penalized linearly, and the fitted function is determined by the training points
# on or outside the $\epsilon$ tube, the *support vectors*. (The margin maximization between classes is the idea of the
# SVM *classifier*, `SVC`.)
#
# The principal-component scores are used without standardization, so the RBF kernel is dominated by the first
# components, which have the largest variance. This works well here; standardizing the scores
# (`sklearn.preprocessing.StandardScaler`) would give all components the same weight.

# %% id="5lTfzpmXPv17"
from sklearn.svm import SVR

# %% colab={"base_uri": "https://localhost:8080/", "height": 80} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1730741828748, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="TfdcdoN2Pv17" outputId="70c836d4-3a45-450b-d80a-01711ed7667f"
# Support vector machine regressor
svr_model = SVR()
svr_model.fit(xtrain, ytrain)

# %% colab={"base_uri": "https://localhost:8080/", "height": 464} executionInfo={"elapsed": 1003, "status": "ok", "timestamp": 1730741829729, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="Pki-5wzePv17" outputId="428e6394-509e-4dd1-960f-53a1361cf305"
plot_score_histograms(svr_model, 'SVM regression')

# %% [markdown]
# ## Evaluation on unseen tubes (leave-one-tube-out)
# The histograms above show the training waveforms themselves, so they do not tell how well the method works for a
# tube that it has never seen. Moreover, the about 100 waveforms of one tube are very similar to each other, so
# splitting the waveforms randomly would also leak information. Therefore, each new or failed tube is held out in
# turn, the PCA and the regression are trained on the other tubes, and the held-out tube is evaluated.
# A tube is classified as failed if the median output of its waveforms exceeds 0.5.

# %%
DECISION_THRESHOLD = 0.5
training_tubes = [
    (class_index, tube_id, waveforms)
    for class_index in (NEW, FAILED)
    for tube_id, waveforms in zip(thyIdList[class_index], dfList[class_index])
]


def leave_one_tube_out(model_factory) -> list[tuple[int, tuple[int, int], np.ndarray]]:
    """Return (class, tube id, outputs of the held-out waveforms) for every held-out new or failed tube."""
    results = []
    for held_out in range(len(training_tubes)):
        rest = [tube for index, tube in enumerate(training_tubes) if index != held_out]
        x_rest = np.vstack([waveforms for _, _, waveforms in rest])
        y_rest = np.hstack([np.full(len(waveforms), float(class_index == FAILED)) for class_index, _, waveforms in rest])
        fold_pca = PCA(n_components=N_COMPONENTS).fit(x_rest)
        model = model_factory().fit(fold_pca.transform(x_rest), y_rest)
        class_index, tube_id, waveforms = training_tubes[held_out]
        results.append((class_index, tube_id, model.predict(fold_pca.transform(waveforms))))
    return results


fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (name, factory) in zip(axes, [("Linear regression", LinearRegression), ("SVM regression", SVR)]):
    results = leave_one_tube_out(factory)
    wrong_tubes = [
        tube_id for class_index, tube_id, outputs in results
        if (np.median(outputs) > DECISION_THRESHOLD) != (class_index == FAILED)
    ]
    wrong_waveforms = sum(int(np.sum((outputs > DECISION_THRESHOLD) != (class_index == FAILED))) for class_index, _, outputs in results)
    total_waveforms = sum(len(outputs) for _, _, outputs in results)
    print(f"{name}: misclassified tubes {len(wrong_tubes)}/{len(results)} {wrong_tubes}, "
          f"misclassified waveforms {wrong_waveforms}/{total_waveforms}")
    for class_index, label in [(NEW, 'held-out new'), (FAILED, 'held-out failed')]:
        outputs = np.hstack([out for c, _, out in results if c == class_index])
        ax.hist(outputs, N_BINS, range=SCORE_RANGE, alpha=0.8, label=label)
    ax.axvline(DECISION_THRESHOLD, color='k', linestyle='--')
    ax.grid()
    ax.legend()
    ax.set_title(f'{name}, leave-one-tube-out', fontsize=14)
    ax.set_xlabel('Regression output')
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Aged tubes versus operating hours
# The output of the model trained on all new and failed tubes is shown for each aged tube (median and interquartile
# range of its waveforms) against the operating hours. A tube whose output approaches 1 has a waveform similar to
# that of the failed tubes and is a candidate for preventive replacement.

# %%
from scipy.stats import spearmanr

aged_hours = np.array([hours for _, hours in thyIdList[AGED]])
fig, ax = plt.subplots(figsize=(8, 5))
for name, model, offset in [("Linear regression", linear_model, -60.), ("SVM regression", svr_model, 60.)]:
    outputs = [model.predict(pca.transform(waveforms)) for waveforms in dfList[AGED]]
    medians = np.array([np.median(out) for out in outputs])
    lower = medians - np.array([np.percentile(out, 25) for out in outputs])
    upper = np.array([np.percentile(out, 75) for out in outputs]) - medians
    ax.errorbar(aged_hours + offset, medians, yerr=[lower, upper], fmt='o', capsize=3, label=name)
    rho, p_value = spearmanr(aged_hours, medians)
    print(f"{name}: Spearman rank correlation with the operating hours rho = {rho:.2f} (p = {p_value:.2f})")
ax.axhline(DECISION_THRESHOLD, color='k', linestyle='--')
ax.set_xlabel('Operating hours')
ax.set_ylabel('Regression output (median, IQR)')
ax.grid()
ax.legend()
plt.show()

# %% [markdown]
# - New and failed tubes are separated also for unseen tubes (leave-one-tube-out above).
# - The aged tubes lie in between. Their output tends to increase with the operating hours, but the correlation is
#   weak: some tubes with fewer hours are already close to the failed tubes, and some old tubes (e.g. S/N 2176 at
#   18100 h) still look new. The waveform carries information on the condition of each tube that the operating hours
#   alone do not provide.
