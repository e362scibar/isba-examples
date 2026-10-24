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
#     display_name: isba
#     language: python
#     name: python3
# ---

# %% [markdown] id="qAe_VWwQ1eoI"
# # Phase-space tomography with CNN
# - Use phase-space distributions with two Gaussian beam components.

# %% [markdown] id="0VPxbIJA_ntB"
# Copyright (C) 2026 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

# %% [markdown] id="o_VeM94r1eoL"
# ## Import modules

# %% id="5Ox7Abm81eoM"
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, gamma, norm, multivariate_normal
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown] id="AjIofieH1eoN"
# ## Parameters
#
# All random numbers are generated from `SEED`, so that the data set and the training are reproducible.

# %% id="f4akWe0T1eoN"
N = 10000   # number of data samples
M = 10000   # number of particles
nbins = 100 # number of bins
xmin = -20.0  # x minimum [mm]
xmax =  20.0  # x maximum [mm]
xpmin = -5.0  # x' minimum [mrad]
xpmax =  5.0  # x' maximum [mrad]
SEED = 0    # random seed
rng = np.random.default_rng(SEED)
xedges = np.linspace(xmin, xmax, nbins+1)
xpedges = np.linspace(xpmin, xpmax, nbins+1)

# %% [markdown] id="8xa79IuT1eoO"
# ## Generate emittance, Twiss parameters, beam centers, randomly
# - Each data sample has two Gaussian components, so two sets of parameters are generated for each sample.

# %% [markdown] id="HYFqs4Wo1eoO"
# ### Distribution of emittance
# - Gamma distribution
# - Geometric emittance ~ 1 $\mu$m rad

# %% id="XbkIdEyi1eoO"
gammarv = gamma(a=5, scale=2.0e-7)
emittance = gammarv.rvs(size=N*2, random_state=rng).reshape(N,2)
fig, ax = plt.subplots()
y,x,h = ax.hist(emittance.reshape(N*2)*1.0e6, bins=100)
ax.set_xlabel(r"emittance [$\mu$m rad]")
plt.show()
print(emittance.min(), emittance.mean())

# %% [markdown] id="NmaAxF1S1eoP"
# ### Distribution of beta
# - Gamma distribution
# - beta ~ 10 m

# %% id="9btrobud1eoP"
gammarv = gamma(a=10, scale=1.0)
beta = gammarv.rvs(size=N*2, random_state=rng).reshape(N,2)
fig, ax = plt.subplots()
y,x,h = ax.hist(beta.reshape(N*2), bins=100)
ax.set_xlabel("beta [m]")
plt.show()
print(beta.min(), beta.mean())

# %% [markdown] id="9z4OUVu81eoQ"
# ### Distribution of alpha
# - Gaussian

# %% id="7EP2Y-u51eoQ"
normrv = norm(loc=0.0, scale=2.0)
alpha = normrv.rvs(size=N*2, random_state=rng).reshape(N,2)
fig, ax = plt.subplots()
y,x,h = ax.hist(alpha.reshape(N*2), bins=100)
ax.set_xlabel("alpha")
plt.show()

# %% [markdown] id="B6LQQtRX1eoQ"
# ### Distribution of beam center
# - 2D Gaussian

# %% id="FP-hGMuA1eoQ"
sig_x = 3.0e-3  # [m]
sig_xp = 3.0e-4  # [rad]
mnormrv = multivariate_normal(mean=np.zeros(2), cov=np.diag([sig_x**2, sig_xp**2]))
center = mnormrv.rvs(size=N*2, random_state=rng).reshape(N,2,2)
z = gaussian_kde(center.reshape(N*2,2).T)(center.reshape(N*2,2).T)
idx = z.argsort()
z2 = z[idx]
fig, ax = plt.subplots()
ax.scatter(center.reshape(N*2,2)[idx,0]*1.0e3, center.reshape(N*2,2)[idx,1]*1.0e3, c=z2, s=1, cmap='rainbow', edgecolor='none')
ax.grid()
ax.set_axisbelow(True)
ax.set_xlabel("x [mm]")
ax.set_ylabel("x' [mrad]")
plt.show()

# %% [markdown] id="gIxmL91u1eoR"
# ## Generate particles

# %% [markdown] id="3HJ_TtAE1eoR"
# ### Calculate covariance matrix
# - `cov[i, k]` is the 2x2 covariance matrix of the component `k` of the sample `i`.

# %% id="7Y6oBULp1eoR"
gam = (1+alpha**2)/beta
cov = np.swapaxes(np.swapaxes(emittance * np.array([[beta, -alpha],[-alpha, gam]]), 0, 2), 1, 3)

# %% [markdown] id="K1Y_pIvN1eoR"
# ### Generate particles
# - Each data sample has two Gaussian components with $M/2$ particles each. The coordinates are in mm and mrad.
# - Storing all particles at once needs $N \times M \times 2$ floats (1.6 GB for $N = M = 10^4$).
#   Instead, the particles of sample $i$ are drawn from a random generator seeded with `(SEED, i)`.
#   The same particles are obtained whenever they are needed, without keeping them in memory.

# %% id="3vFtY4HM1eoR"
def generate_particles(i):
    """Return M particles (x [mm], x' [mrad]) of sample i, shape (M, 2). The same particles are returned for the same i."""
    rng_i = np.random.default_rng([SEED, i])
    return np.concatenate(
        [rng_i.multivariate_normal(center[i,k], cov[i,k], size=M//2) * 1.0e3 for k in range(2)]
    )


# %% [markdown] id="LKAQFMJh1eoR"
# ### Calculate 2D phase-space histogram
# - Each image is normalized to its maximum, so the pixel values are in [0, 1].

# %% id="uPz1FX-Y1eoR"
phasespace = np.empty((N,nbins,nbins))
frac_out_phasespace = np.empty(N)  # fraction of particles outside the phase-space window
for i in range(N):
    particles = generate_particles(i)
    h, _, _ = np.histogram2d(particles[:,0], particles[:,1], bins=(xedges, xpedges))
    frac_out_phasespace[i] = 1.0 - h.sum()/M
    phasespace[i,:,:] = h / np.max(h)

# %% [markdown] id="vbJ6E5qe1eoS"
# ### Check phase-space distribution
# - Emittance, alpha, beta, and gamma are printed for each of the two components.

# %% id="xty6EgaA1eoS"
for i in range(10):
    for k in range(2):
        eps = np.sqrt(np.linalg.det(cov[i,k]))
        a = -cov[i,k,0,1]/eps
        b = cov[i,k,0,0]/eps
        g = cov[i,k,1,1]/eps
        print(f"sample {i}, component {k}: emittance = {eps:.3e} m rad, alpha = {a:+.3f}, beta = {b:.3f} m, gamma = {g:.3f} 1/m")
    data = generate_particles(i)
    z = gaussian_kde(data.T)(data.T)
    idx = z.argsort()
    z2 = z[idx]
    fig, ax = plt.subplots(2,2,figsize=(6,4), gridspec_kw={'height_ratios': [3,1]})
    ax[0,0].scatter(data[idx,0], data[idx,1], c=z2, s=1, cmap='rainbow', edgecolor='none')
    ax[0,0].grid()
    ax[0,0].set_axisbelow(True)
    ax[0,0].set_xlim((xmin,xmax))
    ax[0,0].set_ylim((xpmin,xpmax))
    ax[0,0].set_xlabel("x [mm]")
    ax[0,0].set_ylabel("x' [mrad]")
    ax[0,0].set_title("Scatter plot")
    y,x,h = ax[1,0].hist(data[:,0], bins=nbins, range=(xmin,xmax), histtype='step')
    ax[1,0].set_xlabel("x [mm]")
    ax[1,0].set_xlim((xmin,xmax))
    ax[0,1].pcolormesh(xedges, xpedges, phasespace[i,:,:].T, cmap='rainbow')
    ax[0,1].set_xlabel("x [mm]")
    ax[0,1].set_ylabel("x' [mrad]")
    ax[0,1].set_title("2D Histogram Image")
    y,x,h = ax[1,1].hist(data[:,0], bins=nbins, range=(xmin,xmax), histtype='step')
    ax[1,1].set_xlabel("x [mm]")
    ax[1,1].set_xlim((xmin,xmax))
    plt.tight_layout()
    plt.show()


# %% [markdown] id="L_kOORbr1eoS"
# ## Generate tomography data

# %% [markdown] id="VmZA44TQ1eoS"
# ### Transfer matrices for phase-space rotation
# - The phase space is rotated by the angle $\theta$ in the normalized coordinates $(x/s,\ s x')$:
#   $M = S^{-1} R(\theta) S$ with $S = \mathrm{diag}(1/s, s)$.
# - This corresponds to a beamline with $\beta = s^2$ and $\alpha = 0$ both at the reconstruction point and at the
#   screen, and with the betatron phase advance $\theta$.

# %% id="lfTDVwna1eoS"
def matrot(theta, scale):
    """Transfer matrix that rotates the phase space by theta in the coordinates normalized with scale."""
    mscale = np.array([[1./scale, 0],[0, scale]])
    mrot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return np.linalg.inv(mscale) @ mrot @ mscale


scale = 2.  # s [sqrt(m)], i.e. beta = 4 m

# %% [markdown] id="RtWec_Et1eoS"
# ### Check rotated phase-space distribution

# %% id="f95kqwKW1eoS"
particles = generate_particles(0)
for i in range(4):
    theta = np.pi*0.25*i
    mat = matrot(theta, scale)
    rotdata = mat @ particles.T
    x = rotdata[0,:]
    y = rotdata[1,:]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x2, y2, z2 = x[idx], y[idx], z[idx]
    fig, ax = plt.subplots(2, 1, figsize=(3,4), gridspec_kw={'height_ratios': [3,1]})
    # scatter
    ax[0].scatter(x2, y2, c=z2, s=1, cmap='rainbow', edgecolor='none')
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("x' [mrad]")
    ax[0].set_title(f"{i*45} deg.")
    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[0].set_xlim((xmin,xmax))
    ax[0].set_ylim((xpmin,xpmax))
    # x projection
    y,x,p = ax[1].hist(rotdata[0,:], bins=nbins, range=(xmin,xmax), histtype='step')
    ax[1].set_xlabel("x [mm]")
    ax[1].set_xlim((xmin,xmax))
    plt.tight_layout()
    plt.show()

# %% [markdown] id="-_9tfqYz1eoT"
# ### Calculate tomography data
# - The rotation angle is scanned from 0 to 178 degrees in 2-degree steps (90 angles).
# - Each tomography image is normalized to its maximum.

# %% id="7Cfp2kwV1eoT"
ANGLE_STEP = 2.0  # [deg]
angles = np.deg2rad(np.arange(0., 180., ANGLE_STEP))
angle_edges = np.arange(-0.5*ANGLE_STEP, 180., ANGLE_STEP)  # [deg], bin edges for the plots
rot_mats = np.array([matrot(theta, scale) for theta in angles])  # (angle, 2, 2)
tomodata = np.empty((N,len(angles),nbins))
frac_out_screen = np.empty(N)  # largest fraction of particles outside the screen over all angles
for j in range(N):
    x_rot = (rot_mats @ generate_particles(j).T)[:,0,:]  # (angle, M)
    for i in range(len(angles)):
        tomodata[j,i,:] = np.histogram(x_rot[i], xedges)[0]
    frac_out_screen[j] = 1.0 - tomodata[j].sum(axis=1).min()/M
    tomodata[j,:,:] /= np.max(tomodata[j,:,:])

# %% [markdown]
# ### Particles outside the window
# The phase-space images cover $\pm 20$ mm and $\pm 5$ mrad, and the screen covers $\pm 20$ mm. For a small fraction of
# the samples, part of the beam is outside this range.

# %%
print(f"samples with >1% of particles outside the phase-space window: {np.mean(frac_out_phasespace > 0.01):.1%}")
print(f"samples with >1% of particles outside the screen at some angle: {np.mean(frac_out_screen > 0.01):.1%}")

# %% [markdown] id="6BQetVAg1eoT"
# ### Check generated images

# %% id="V7vMX8Kg1eoT"
for i in range(10):
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    ax[0].pcolormesh(xedges, angle_edges, tomodata[i,:,:], cmap='rainbow')
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("angle [deg.]")
    ax[0].set_title("Tomography")
    ax[1].pcolormesh(xedges, xpedges, phasespace[i,:,:].T, cmap='rainbow')
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel("x' [mrad]")
    ax[1].set_xlim((xmin,xmax))
    ax[1].set_ylim((xpmin,xpmax))
    ax[1].set_title("Phase-space")
    plt.tight_layout()
    plt.show()

# %% [markdown] id="qsw26Tzz1eoT"
# ## Prepare data for CNN

# %% [markdown] id="inErkKnL1eoU"
# ### Split data into training, validation, and test
# - Training data: used to update the CNN parameters.
# - Validation data: used to monitor the loss during the training.
# - Test data: not used during the training at all; used only for the final evaluation.

# %% id="feZK_7MM1eoU"
n_train, n_val, n_test = 6000, 2000, 2000
x_train, y_train = tomodata[:n_train,:,:], phasespace[:n_train,:]
x_val, y_val = tomodata[n_train:n_train+n_val,:,:], phasespace[n_train:n_train+n_val,:]
x_test, y_test = tomodata[-n_test:,:,:], phasespace[-n_test:,:]
test_offset = N - n_test  # index of the first test sample in the whole data set

# %% [markdown] id="j2k9T_BG1eoU"
# ### Input and output image dimensions
# - Input: 90 (angle) x 100 (x) pixels
# - Output: 100 (x) x 100 (x') pixels

# %% id="BD4yPIHk1eoU"
input_shape = (len(angles), nbins, 1)
output_shape = (nbins, nbins, 1)

# %% [markdown] id="ParfxQ3C1eoY"
# ### Reshape data
# - The arrays already have the image layout (sample, row, column), so only a channel axis is added.

# %% id="c0jbT8ad1eoY"
x_train1 = x_train[..., np.newaxis].astype(np.float32)
x_val1 = x_val[..., np.newaxis].astype(np.float32)
x_test1 = x_test[..., np.newaxis].astype(np.float32)
y_train1 = y_train[..., np.newaxis].astype(np.float32)
y_val1 = y_val[..., np.newaxis].astype(np.float32)
y_test1 = y_test[..., np.newaxis].astype(np.float32)

# %% [markdown] id="7KihZuYx1eoY"
# ## Training of CNN

# %% [markdown] id="6sPJPhWR1eoY"
# ### Import modules

# %% id="MV1iS1E_1eoY"
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

keras.utils.set_random_seed(SEED)
batch_size = 256
epochs = 100  # about 1000 epochs reduce the loss slightly further, but take much longer

# %% [markdown] id="J_8hDqnP1eoa"
# ### Make CNN model
# - The loss function is the binary cross-entropy averaged over the pixels,
#   $-\left[ y \ln \hat{y} + (1 - y) \ln (1 - \hat{y}) \right]$, where $y$ and $\hat{y}$ are the true and predicted
#   pixel values in [0, 1]. Its minimum is not zero for $0 < y < 1$, so the loss value itself is not an error measure.

# %% id="5mh_9B_m1eoa"
model = Sequential()
model.add(Input(input_shape))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(10, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(25*25, activation='relu'))
model.add(Reshape((25,25,1)))
model.add(Conv2D(10, kernel_size=(3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# %% [markdown] id="3aqF2myJ1eoa"
# ### Check model

# %% id="vDVQQVgp1eob"
model.summary()

# %% [markdown] id="tTvGpW1O1eob"
# ### Fit the model to data

# %% id="7Tlp5TD-1eob"
history = model.fit(x_train1, y_train1, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val1, y_val1))
score = model.evaluate(x_test1, y_test1, verbose=0)
print('Test loss (binary cross-entropy):', score)

# %% [markdown] id="lPu0v5xC1eob"
# ### Evolution of loss function

# %% id="2lzfhXWi1eob"
fig, ax = plt.subplots()
ax.semilogy(history.history['loss'], label='Training loss')
ax.semilogy(history.history['val_loss'], label='Validation loss')
ax.set_title('Loss')
ax.set_xlabel('Epoch')
ax.grid()
ax.legend()
plt.show()

# %% [markdown] id="XCQnRNKe1eob"
# ## Check results with the test data

# %% id="EfpWKzOU1eob"
predictions = model.predict(x_test1)[..., 0]

# %% [markdown]
# ### Beam parameters from the phase-space images
# - The rms emittance and the Twiss parameters are calculated from the second moments of each image, with the pixel
#   values as weights, for both the prediction and the ground truth.
# - Note that these are the parameters of the whole two-component beam, not of each component.
# - The second moments are sensitive to a faint background far from the beam core, so a small background in the
#   predicted image can enlarge the rms emittance considerably even if the image looks similar.

# %%
xcenters = 0.5*(xedges[1:] + xedges[:-1])
xpcenters = 0.5*(xpedges[1:] + xpedges[:-1])
X, XP = np.meshgrid(xcenters, xpcenters, indexing='ij')  # same layout as the images (x, x')


def image_twiss(img):
    """Return the rms emittance [um rad], alpha, and beta [m] of a phase-space image in (x [mm], x' [mrad])."""
    w = img / img.sum()
    dx = X - (w*X).sum()
    dxp = XP - (w*XP).sum()
    sxx, sxpxp, sxxp = (w*dx*dx).sum(), (w*dxp*dxp).sum(), (w*dx*dxp).sum()
    eps = np.sqrt(sxx*sxpxp - sxxp**2)
    return np.array([eps, -sxxp/eps, sxx/eps])


twiss_pred = np.array([image_twiss(img) for img in predictions])
twiss_truth = np.array([image_twiss(img) for img in y_test])
error = twiss_pred - twiss_truth
summary = pd.DataFrame(
    {
        "RMSE": np.sqrt(np.mean(error**2, axis=0)),
        "median |relative error|": [
            np.median(np.abs(error[:,0]/twiss_truth[:,0])), np.nan, np.median(np.abs(error[:,2]/twiss_truth[:,2]))
        ],
    },
    index=["rms emittance [um rad]", "alpha", "beta [m]"],
)
print(f"image MSE (pixel values in [0, 1]): {np.mean((predictions - y_test)**2):.3e}")
summary

# %% [markdown]
# ### Spot check of phase-space distribution

# %%
for i in range(10):
    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].pcolormesh(xedges, angle_edges, tomodata[test_offset+i,:,:], cmap='rainbow')
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("angle [deg.]")
    ax[0].set_title("Tomography")
    ax[1].pcolormesh(xedges, xpedges, predictions[i,:,:].T, cmap='rainbow')
    ax[1].set_xlim((xmin,xmax))
    ax[1].set_ylim((xpmin,xpmax))
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel("x' [mrad]")
    ax[1].set_title("Prediction")
    ax[2].pcolormesh(xedges, xpedges, phasespace[test_offset+i,:,:].T, cmap='rainbow')
    ax[2].set_xlim((xmin,xmax))
    ax[2].set_ylim((xpmin,xpmax))
    ax[2].set_xlabel("x [mm]")
    ax[2].set_ylabel("x' [mrad]")
    ax[2].set_title("Ground Truth")
    plt.tight_layout()
    plt.show()
