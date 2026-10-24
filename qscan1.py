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

# %% [markdown] id="gZ2i1pK0wL7R"
# # Regression of emittance and Twiss parameters from quadrupole scan data
# - Use Gaussian beam
#

# %% [markdown] id="wok7GUis_iWw"
#
# Copyright (C) 2026 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

# %% [markdown] id="qbYcWv1DwL7U"
# ## Import modules

# %% id="tY8b3mTVwL7V"
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, gamma
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown] id="wbkna0PQwL7X"
#
# ## Parameters
#
# All random numbers are generated from `SEED`, so that the data set and the training are reproducible.

# %% id="EQI1TRVWwL7X"
N = 10000   # number of data samples
M = 10000   # number of particles
L = 5.0     # drift length [m]
f0inv = 1.0 # 1/f_0 (reciprocal of minimum focal length of the Q) [1/m]
f1inv = 0.0 # 1/f_1 (reciprocal of maximum focal length of the Q) [1/m]
NQ = 21     # number of Q settings
SEED = 0    # random seed
rng = np.random.default_rng(SEED)

# %% [markdown] id="y8fijaKewL7X"
# ## Generate emittance and Twiss parameters randomly

# %% [markdown] id="9WIikJs-wL7Y"
# ### Distribution of emittance
# - Gamma distribution
# - Geometric emittance ~ 5 $\mu$m rad

# %% id="g-myY06GwL7Y"
gammarv = gamma(a=5, scale=1.0e-6)
emittance = gammarv.rvs(size=N, random_state=rng)
fig, ax = plt.subplots()
y,x,h = ax.hist(emittance*1.0e6, bins=100)
ax.set_xlabel(r"emittance [$\mu$m rad]")
plt.show()
print(emittance.min(), emittance.mean())

# %% [markdown] id="04Zc9-ATwL7Z"
# ### Distribution of beta
# - Gamma distribution
# - beta ~ 10 m

# %% id="Rw_OfPDSwL7Z"
gammarv = gamma(a=10, scale=1.0)
beta = gammarv.rvs(size=N, random_state=rng)
fig, ax = plt.subplots()
y,x,h = ax.hist(beta, bins=100)
ax.set_xlabel("beta [m]")
plt.show()
print(beta.min(), beta.mean())

# %% [markdown] id="xdXX2o2RwL7Z"
# ### Distribution of alpha
# - uniform distribution within the boundary
# - The boundary condition is the existence of a beam waist on the screen within the scan range.
# - The beam size is minimum at $1/f = 1/L - \alpha/\beta$, so $1/f_1 \le 1/f \le 1/f_0$ gives
#   $\beta(1/L - 1/f_0) < \alpha < \beta(1/L - 1/f_1)$

# %% id="XwyesbPHwL7a"
a0, a1 = beta*(1.0/L - f0inv), beta*(1.0/L - f1inv)
urand = rng.uniform(size=N)
alpha = urand*(a1-a0) + a0
fig, ax = plt.subplots()
y,x,h = ax.hist(alpha, bins=100)
ax.set_xlabel("alpha")
plt.show()

# %% [markdown] id="BswHfIj2wL7a"
# ## Generate particles

# %% [markdown] id="nYQ_EmndwL7a"
# ### Calculate covariance matrix

# %% id="_wah7LYawL7a"
gam = (1+alpha**2)/beta
cov = np.rollaxis(emittance * np.array([[beta, -alpha],[-alpha, gam]]), -1)

# %% [markdown] id="tWVT-yvbwL7a"
# ### Generate particles
# - Storing all particles at once needs $N \times M \times 2$ floats (1.6 GB for $N = M = 10^4$).
# - Instead, the particles of sample $i$ are drawn from a random generator seeded with `(SEED, i)`.
#   The same particles are obtained whenever they are needed, without keeping them in memory.

# %% id="5j1HHuqswL7a"
def generate_particles(i, cov_i=None):
    """Return M particles (x, x') of sample i, shape (M, 2).

    The same particles are returned for the same i. If cov_i is given, it is used instead of the covariance matrix
    of sample i (used for the predicted distributions).
    """
    rng_i = np.random.default_rng([SEED, i])
    return rng_i.multivariate_normal(np.zeros(2), cov[i] if cov_i is None else cov_i, size=M)


# %% [markdown] id="7AfmX3uwwL7a"
# ### Check distribution

# %% id="zovVPZRWwL7b"
for i in range(10):
    eps = np.sqrt(np.linalg.det(cov[i,:,:]))
    a = -cov[i,0,1]/eps
    b = cov[i,0,0]/eps
    g = cov[i,1,1]/eps
    print(i, eps, a, b, g)
    data = generate_particles(i)
    z = gaussian_kde(data.T)(data.T)
    idx = z.argsort()
    z2 = z[idx]
    fig, ax = plt.subplots(figsize=(3,3))
    ax.scatter(data[idx,0]*1.0e3, data[idx,1]*1.0e3, c=z2, s=1, cmap='rainbow', edgecolor='none')
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlim((-30,30))
    ax.set_ylim((-10,10))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("x' [mrad]")
    plt.show()

# %% [markdown] id="xQ2C7yEvwL7b"
# ## Generate quadrupole scan data

# %% [markdown] id="sNn_oMVFwL7b"
# ### Calculate transfer matrices of Q-scan
# - Thin-lens quadrupole followed by a drift: $M = \begin{pmatrix} 1 & L \\ 0 & 1 \end{pmatrix}
#   \begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$
# - The Q settings are ordered from $1/f_0$ (strongest) to $1/f_1$ (weakest).

# %% id="dwWHxWbHwL7b"
finv = np.linspace(f0inv, f1inv, NQ)
qmat = np.ones((NQ,2,2))
qmat[:,0,1] = 0.
qmat[:,1,0] = -finv
drift = np.eye(2)
drift[0,1] = L
tmat = drift @ qmat

# %% [markdown] id="ZnWire7wwL7b"
# ### Generate Q-scan data
# - `qscan[j, i, :]` is the horizontal beam profile of sample `j` at the Q setting `finv[i]`.

# %% id="yWa5EQ9qwL7b"
Nbin = 60
Xmin = -30.0e-3
Xmax =  30.0e-3
x_edges = np.linspace(Xmin, Xmax, Nbin+1)
qscan = np.empty((N, NQ, Nbin))
for j in range(N):
    p1 = tmat @ generate_particles(j).T  # (NQ, 2, M)
    for i in range(NQ):
        qscan[j,i,:] = np.histogram(p1[i,0,:], x_edges)[0]


# %% [markdown]
# ### Particles outside the screen
# The screen covers only $\pm 30$ mm. Near the ends of the scan range the beam can be much larger than this, and part
# of the profile is lost. The images still carry this information (the total intensity drops), but large-$\beta$
# beams and beams whose waist is at the end of the scan range are harder to predict.

# %%
frac_inside = qscan.sum(axis=2) / M  # (N, NQ)
clipped = frac_inside.min(axis=1) < 0.99
print(f"samples with >1% of particles outside the screen at some Q setting: {clipped.mean():.1%}")
print(f"  for beta > 15 m: {clipped[beta > 15].mean():.1%}")

# %% [markdown] id="qNNiEYOKwL7b"
# ### Check Q-scan data
# - The dashed line shows the analytic waist position $1/f = 1/L - \alpha/\beta$.

# %% id="ndSbqGIawL7c"
df = (f0inv-f1inv)/(NQ-1)
finv_edges = np.linspace(f0inv+df*0.5, f1inv-df*0.5, NQ+1)  # same order as the rows of qscan
for i in range(10):
    fig, ax = plt.subplots(figsize=(3,2))
    ax.pcolormesh(x_edges*1.e3, finv_edges, qscan[i,:,:], cmap='rainbow')
    ax.axhline(1.0/L - alpha[i]/beta[i], color='white', linestyle='--', linewidth=1.0)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("1/f [1/m]")
    plt.show()

# %% [markdown] id="vAvpyUsrwL7c"
# ## Prepare data for CNN

# %% [markdown] id="06uIa0ZuwL7c"
# ### Split data into training, validation, and test
# - Training data: used to update the CNN parameters.
# - Validation data: used to monitor the loss during the training.
# - Test data: not used during the training at all; used only for the final evaluation.
#
# The targets (emittance in $\mu$m rad, alpha, and beta in m) have different scales. Each of them is standardized
# with the mean and standard deviation of the training data, so that the three parameters contribute equally to the
# mean squared error.

# %% id="J3H2YVy4wL7c"
n_train, n_val, n_test = 6000, 2000, 2000
y_data = np.array([emittance*1.0e6, alpha, beta]).T
y_mean = y_data[:n_train].mean(axis=0)
y_std = y_data[:n_train].std(axis=0)
y_scaled = (y_data - y_mean) / y_std
x_train, y_train = qscan[:n_train,:,:], y_scaled[:n_train,:]
x_val, y_val = qscan[n_train:n_train+n_val,:,:], y_scaled[n_train:n_train+n_val,:]
x_test, y_test = qscan[-n_test:,:,:], y_scaled[-n_test:,:]
test_offset = N - n_test  # index of the first test sample in the whole data set

# %% [markdown] id="jUhPQXanwL7c"
# ### Input image dimensions
# - Rows: x bins, columns: Q settings

# %% id="IiREwYZ3wL7c"
img_rows, img_cols = Nbin, NQ
input_shape = (img_rows, img_cols, 1)

# %% [markdown] id="f2TwkOuuwL7c"
# ### Reshape and scale input data
# - `qscan` has the shape (sample, Q setting, x bin). It is transposed (not reshaped) to (sample, x bin, Q setting),
#   so that neighboring pixels of the image are neighboring x bins and Q settings.

# %% id="Ek9wEXU4wL7c"
def to_image(q):
    """Convert Q-scan data of shape (n, NQ, Nbin) into CNN input images of shape (n, Nbin, NQ, 1)."""
    return np.transpose(q, (0, 2, 1))[..., np.newaxis].astype(np.float32) / M


x_train1 = to_image(x_train)
x_val1 = to_image(x_val)
x_test1 = to_image(x_test)

# %% [markdown] id="dWNGaF7vwL7d"
# ## Training of CNN

# %% [markdown] id="wvrXLgYfwL7d"
# ### Import modules

# %% id="enPYDwPcwL7d"
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

keras.utils.set_random_seed(SEED)
batch_size = 256
epochs = 30

# %% [markdown] id="N34YyCt4wL7d"
# ### Make CNN model

# %% id="tYWoGFJZwL7d"
# Make model
model = Sequential()
model.add(Input(input_shape))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# %% [markdown] id="VwuYywO0wL7d"
# ### Check model

# %% id="zw7bmqjPwL7h"
model.summary()

# %% [markdown] id="0WfvoVXGwL7i"
# ### Fit the model to data (Training)

# %% id="Y06_NgZIwL7i"
history = model.fit(x_train1, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val1, y_val))
score = model.evaluate(x_test1, y_test, verbose=0)
print('Test loss (MSE of standardized targets):', score[0])
print('Test MAE (standardized targets):', score[1])

# %% [markdown] id="7Y9-yxbAwL7i"
# ### Evolution of loss function

# %% id="861vDJMTwL7i"
fig, ax = plt.subplots()
ax.semilogy(history.history['loss'], label='Training loss')
ax.semilogy(history.history['val_loss'], label='Validation loss')
ax.set_title('Loss')
ax.set_xlabel('Epoch')
ax.grid()
ax.legend()
plt.show()

# %% [markdown] id="zCdes0VzwL7i"
# ## Check results with the test data

# %% [markdown] id="s9CW29LswL7i"
# ### Spot check

# %% id="I34dCi12wL7i"
pred = model.predict(x_test1) * y_std + y_mean
truth = y_data[-n_test:]
print('            emittance [um rad],      alpha,   beta [m]')
for i in range(10):
    print('prediction  ', pred[i])
    print('ground truth', truth[i])
    print('error       ', pred[i] - truth[i])
    print()

# %% [markdown]
# ### Prediction errors
# - RMSE and the median of the relative error for each parameter. The relative error of alpha is not shown because
#   alpha can be close to zero.

# %%
error = pred - truth
summary = pd.DataFrame(
    {
        "RMSE": np.sqrt(np.mean(error**2, axis=0)),
        "median |relative error|": [np.median(np.abs(error[:,0]/truth[:,0])), np.nan, np.median(np.abs(error[:,2]/truth[:,2]))],
    },
    index=["emittance [um rad]", "alpha", "beta [m]"],
)
summary

# %% [markdown] id="HhxpRoauwL7j"
# ### Compare phase-space distribution

# %% id="yg_nOdE9wL7j"
for i in range(10):
    data = generate_particles(test_offset+i)
    eps, a, b = pred[i]
    if eps <= 0.0 or b <= 0.0:
        print(f"test sample {i}: unphysical prediction (emittance = {eps:.3g}, beta = {b:.3g}), skipped")
        continue
    g = (1+a**2)/b
    cov_pred = eps*1.0e-6 * np.array([[b, -a],[-a, g]])
    pred_particles = generate_particles(test_offset+i, cov_pred)
    z = gaussian_kde(data.T)(data.T)
    idx = z.argsort()
    z2 = z[idx]
    z = gaussian_kde(pred_particles.T)(pred_particles.T)
    idx2 = z.argsort()
    z3 = z[idx2]
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    ax[0].scatter(data[idx,0]*1.0e3, data[idx,1]*1.0e3, c=z2, s=1, cmap='rainbow', edgecolor='none')
    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[0].set_xlim((-30,30))
    ax[0].set_ylim((-10,10))
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("x' [mrad]")
    ax[0].set_title("Ground Truth")
    ax[1].scatter(pred_particles[idx2,0]*1.0e3, pred_particles[idx2,1]*1.0e3, c=z3, s=1, cmap='rainbow', edgecolor='none')
    ax[1].grid()
    ax[1].set_axisbelow(True)
    ax[1].set_xlim((-30,30))
    ax[1].set_ylim((-10,10))
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel("x' [mrad]")
    ax[1].set_title("Prediction")
    plt.tight_layout()
    plt.show()

# %% [markdown] id="S4WVfY4NwL7j"
# ### Compare emittance and Twiss parameters

# %% id="FQHc6lEYwL7j"
pred_emittance = pred[:,0]
pred_alpha = pred[:,1]
pred_beta = pred[:,2]
truth_emittance = truth[:,0]
truth_alpha = truth[:,1]
truth_beta = truth[:,2]
# Emittance
xy = np.vstack([truth_emittance, pred_emittance])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
z2 = z[idx]
fig, ax = plt.subplots(figsize=(3,3))
ax.scatter(truth_emittance[idx], pred_emittance[idx], c=z2, s=1, cmap='rainbow', edgecolor='none')
ax.grid()
ax.set_axisbelow(True)
ax.set_xlabel(r"Truth [$\mu$m rad]")
ax.set_ylabel(r"Prediction [$\mu$m rad]")
ax.set_title("Emittance")
ax.set_xlim((0.,None))
ax.set_ylim((0.,None))
plt.show()
# Alpha
xy = np.vstack([truth_alpha, pred_alpha])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
z2 = z[idx]
fig, ax = plt.subplots(figsize=(3,3))
ax.scatter(truth_alpha[idx], pred_alpha[idx], c=z2, s=1, cmap='rainbow', edgecolor='none')
ax.grid()
ax.set_axisbelow(True)
ax.set_xlabel("Truth")
ax.set_ylabel("Prediction")
ax.set_title("Alpha")
plt.show()
# Beta
xy = np.vstack([truth_beta, pred_beta])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
z2 = z[idx]
fig, ax = plt.subplots(figsize=(3,3))
ax.scatter(truth_beta[idx], pred_beta[idx], c=z2, s=1, cmap='rainbow', edgecolor='none')
ax.grid()
ax.set_axisbelow(True)
ax.set_xlabel("Truth [m]")
ax.set_ylabel("Prediction [m]")
ax.set_title("Beta")
ax.set_xlim((0.,None))
ax.set_ylim((0.,None))
plt.show()
