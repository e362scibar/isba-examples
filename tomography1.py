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
#     name: python3
# ---

# %% [markdown] id="qAe_VWwQ1eoI"
# # Phase-space tomography with CNN
# - Use phase-space distributions with two Gaussian beam components.

# %% [markdown] id="0VPxbIJA_ntB"
# Copyright (C) 2024 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
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
import math
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown] id="AjIofieH1eoN"
# ## Parameters

# %% id="f4akWe0T1eoN"
N = 10000   # number of data samples
M = 10000   # number of particles
nbins = 100 # number of bins
xmin = -20.0  # x minimum [mm]
xmax =  20.0  # x maximum [mm]
xpmin = -5.0  # x' minimum [mrad]
xpmax =  5.0  # x' maximum [mrad]

# %% [markdown] id="8xa79IuT1eoO"
# ## Generate emittance, Twiss parameters, beam centers, randomly

# %% [markdown] id="HYFqs4Wo1eoO"
# ### Distribution of emittance
# - Gamma distribution
# - natual emittance ~ 1 $\mu$ m

# %% colab={"base_uri": "https://localhost:8080/", "height": 470} id="XbkIdEyi1eoO" executionInfo={"status": "ok", "timestamp": 1730892781681, "user_tz": -420, "elapsed": 735, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="902783f4-ad36-4717-dab2-3112625812c8"
gammarv = gamma(a=5, scale=2.0e-7)
emittance = gammarv.rvs(size=N*2).reshape(N,2)
fig, ax = plt.subplots()
y,x,h = ax.hist(emittance.reshape(N*2)*1.0e6, bins=100)
ax.set_xlabel("emittance [$\mu$m rad]")
plt.show()
print(emittance.min(), emittance.mean())

# %% [markdown] id="NmaAxF1S1eoP"
# ### Distribution of beta
# - Gamma distribution
# - beta ~ 10 m

# %% colab={"base_uri": "https://localhost:8080/", "height": 466} id="9btrobud1eoP" executionInfo={"status": "ok", "timestamp": 1730892782478, "user_tz": -420, "elapsed": 844, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="2a709fbc-283f-4812-d8a3-d3e2f4f7e1c3"
gammarv = gamma(a=10, scale=1.0)
beta = gammarv.rvs(size=N*2).reshape(N,2)
fig, ax = plt.subplots()
y,x,h = ax.hist(beta.reshape(N*2), bins=100)
ax.set_xlabel("beta [m]")
plt.show()
print(beta.min(), beta.mean())

# %% [markdown] id="9z4OUVu81eoQ"
# ### Distribution of alpha
# - Gaussian

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} id="7EP2Y-u51eoQ" executionInfo={"status": "ok", "timestamp": 1730892783075, "user_tz": -420, "elapsed": 619, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="290884e7-68c6-4c3a-94da-f191d04fdf85"
normrv = norm(loc=0.0, scale=2.0)
alpha = normrv.rvs(size=N*2).reshape(N,2)
fig, ax = plt.subplots()
y,x,h = ax.hist(alpha.reshape(N*2), bins=100)
ax.set_xlabel("alpha")
plt.show()

# %% [markdown] id="B6LQQtRX1eoQ"
# ### Distribution of beam center
# - 2D Gaussian

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} id="FP-hGMuA1eoQ" executionInfo={"status": "ok", "timestamp": 1730892797120, "user_tz": -420, "elapsed": 14064, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="ad207a81-7a0d-4d10-cf65-7cfaa6fc8f20"
sig_x = 3.0e-3  # [m]
sig_xp = 3.0e-4  # [m]
mnormrv = multivariate_normal(mean=np.zeros(2), cov=np.diag([sig_x**2, sig_xp**2]))
center = mnormrv.rvs(size=N*2).reshape(N,2,2)
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

# %% id="7Y6oBULp1eoR"
gam = (1+alpha**2)/beta
cov = np.swapaxes(np.swapaxes(emittance * np.array([[beta, -alpha],[-alpha, gam]]), 0, 2), 1, 3)

# %% [markdown] id="K1Y_pIvN1eoR"
# ### Generate particles
# - Each data has two Gaussian components

# %% id="3vFtY4HM1eoR"
particles = np.empty((N,M,2))
for i in range(N):
    particles[i,:M//2,:] = multivariate_normal.rvs(mean=center[i,0], cov=cov[i,0,:,:], size=M//2) * 1.0e3
    particles[i,M//2:,:] = multivariate_normal.rvs(mean=center[i,1], cov=cov[i,1,:,:], size=M//2) * 1.0e3

# %% [markdown] id="LKAQFMJh1eoR"
# ### Calculate 2D phase-space histogram

# %% id="uPz1FX-Y1eoR"
phasespace = np.empty((N,nbins,nbins))
for i in range(N):
    h, x, y = np.histogram2d(particles[i,:,0], particles[i,:,1], bins=(nbins,nbins), range=((xmin,xmax),(xpmin,xpmax)))
    phasespace[i,:,:] = h / np.max(h)
xedges = x
yedges = y

# %% [markdown] id="vbJ6E5qe1eoS"
# ### Check phase-space distribution

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="xty6EgaA1eoS" executionInfo={"status": "ok", "timestamp": 1730892840571, "user_tz": -420, "elapsed": 20217, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="7ff64ca5-f309-4c9d-bd2f-de2034bdded5"
for i in range(10):
    eps = np.sqrt(np.linalg.det(cov[i,:,:]))
    a = -cov[i,0,1]/eps
    b = cov[i,0,0]/eps
    g = cov[i,1,1]/eps
    print(i, eps, a, b, g)
    data = particles[i,:,:]
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
    y,x,h = ax[1,0].hist(particles[i,:,0], bins=nbins, range=(xmin,xmax), histtype='step')
    ax[1,0].set_xlabel("x [mm]")
    ax[1,0].set_xlim((xmin,xmax))
    ax[0,1].pcolormesh(xedges, yedges, phasespace[i,:,:].T, cmap='rainbow')
    ax[0,1].set_xlabel("x [mm]")
    ax[0,1].set_ylabel("x' [mrad]")
    ax[0,1].set_title("2D Histogram Image")
    y,x,h = ax[1,1].hist(particles[i,:,0], bins=nbins, range=(xmin,xmax), histtype='step')
    ax[1,1].set_xlabel("x [mm]")
    ax[1,1].set_xlim((xmin,xmax))
    plt.tight_layout()
    plt.show()


# %% [markdown] id="L_kOORbr1eoS"
# ## Generate tomography data

# %% [markdown] id="VmZA44TQ1eoS"
# ### Transfer matorices for phase-space rotation

# %% id="lfTDVwna1eoS"
def matrot(theta, scale):
    mscale = np.array([[1./scale, 0],[0, scale]])
    mrot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return np.linalg.inv(mscale) @ mrot @ mscale


# %% [markdown] id="RtWec_Et1eoS"
# ### Check rotated phase-space distribution

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="f95kqwKW1eoS" executionInfo={"status": "ok", "timestamp": 1730892847093, "user_tz": -420, "elapsed": 6525, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="5c6d16b1-6f6f-45fa-bb3f-5e5df62af210"
scale = 2.
for i in range(4):
    theta = np.pi*0.25*i
    mat = matrot(theta, scale)
    rotdata = mat @ particles[0,:,:].T
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
    del rotdata,x,y,xy,z,idx,x2,y2,z2,fig,ax

# %% [markdown] id="-_9tfqYz1eoT"
# ### Calculate tomography data

# %% colab={"base_uri": "https://localhost:8080/"} id="7Cfp2kwV1eoT" executionInfo={"status": "ok", "timestamp": 1730893096481, "user_tz": -420, "elapsed": 80226, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="4c56358e-6d69-49e2-8b7d-425e6598e7e7"
scale = 2.
angles = np.deg2rad(np.arange(0.,180.,2.)) # 2-deg step
tomodata = np.empty((N,len(angles),nbins))
for i,theta in enumerate(angles):
    if i%5 == 0:
        print(np.round(np.rad2deg(theta)))
    mat = matrot(theta, scale)
    rotdata = np.swapaxes(mat @ np.swapaxes(particles, 1, 2), 1, 2)
    for j in range(N):
        val, edge = np.histogram(rotdata[j,:,0], nbins, (xmin,xmax))
        tomodata[j,i,:] = val / M
    del rotdata
print("Done.")
for i in range(N):
    tomodata[i,:,:] /= np.max(tomodata[i,:,:])
print("Normalization done.")

# %% [markdown] id="6BQetVAg1eoT"
# ### Check generated images

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="V7vMX8Kg1eoT" executionInfo={"status": "ok", "timestamp": 1730893100774, "user_tz": -420, "elapsed": 4300, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="f4bd5cae-6145-418e-a89d-bd6260db44e9"
for i in range(10):
    print(i)
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    ax[0].pcolormesh(np.linspace(xmin,xmax,nbins+1), np.linspace(-1.,181.,91), tomodata[i,:,:], cmap='rainbow')
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("angle [deg.]")
    ax[0].set_title("Tomography")
    ax[1].pcolormesh(xedges, yedges, phasespace[i,:,:].T, cmap='rainbow')
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel("x' [mrad]")
    ax[1].set_xlim((xmin,xmax))
    ax[1].set_ylim((xpmin,xpmax))
    ax[1].set_title("Phase-space")
    plt.tight_layout()
    plt.show()
    print(np.max(tomodata[i,:,:]))

# %% [markdown] id="qsw26Tzz1eoT"
# ## Prepare data for CNN

# %% [markdown] id="inErkKnL1eoU"
# ### Split data into training, test, and validation

# %% id="feZK_7MM1eoU"
n_train, n_test, n_val = 6000, 2000, 2000
x_train, y_train = tomodata[:n_train,:,:], phasespace[:n_train,:]
x_test, y_test = tomodata[n_train:n_train+n_test,:,:], phasespace[n_train:n_train+n_test,:]
x_val, y_val = tomodata[-n_val:,:,:], phasespace[-n_val:,:]

# %% [markdown] id="j2k9T_BG1eoU"
# ### Input and output image dimensions

# %% id="BD4yPIHk1eoU"
input_shape = (len(angles), nbins, 1)
output_shape = (nbins, nbins, 1)

# %% [markdown] id="ParfxQ3C1eoY"
# ### Reshape data

# %% id="c0jbT8ad1eoY"
x_train1 = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)
x_test1 = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)
x_val1 = x_val.reshape(x_val.shape[0], input_shape[0], input_shape[1], input_shape[2]).astype(np.float32)
y_train1 = y_train.reshape(y_train.shape[0], output_shape[0], output_shape[1], output_shape[2]).astype(np.float32)
y_test1 = y_test.reshape(y_test.shape[0], output_shape[0], output_shape[1], output_shape[2]).astype(np.float32)
y_val1 = y_val.reshape(y_val.shape[0], output_shape[0], output_shape[1], output_shape[2]).astype(np.float32)

# %% [markdown] id="7KihZuYx1eoY"
# ## Training of CNN

# %% [markdown] id="6sPJPhWR1eoY"
# ### Import modules

# %% id="MV1iS1E_1eoY"
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K

batch_size = 128
#epochs = 1000
epochs = 100

# %% [markdown] id="J_8hDqnP1eoa"
# ### Make CNN model

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

# %% colab={"base_uri": "https://localhost:8080/", "height": 562} id="vDVQQVgp1eob" executionInfo={"status": "ok", "timestamp": 1730893103876, "user_tz": -420, "elapsed": 9, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="0ea47b8a-aa26-4eee-dbd5-d5a393e04cc9"
model.summary()

# %% [markdown] id="tTvGpW1O1eob"
# ### Fit the model to data

# %% colab={"base_uri": "https://localhost:8080/"} id="7Tlp5TD-1eob" executionInfo={"status": "ok", "timestamp": 1730893431214, "user_tz": -420, "elapsed": 138783, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="3786fdbe-a44e-4778-93e8-623fe73ac036"
history = model.fit(x_train1, y_train1, batch_size=2*batch_size, epochs=epochs, verbose=1, validation_data=(x_test1, y_test1))
score = model.evaluate(x_test1, y_test1, verbose=0)
print('Test loss:', score)

# %% [markdown] id="lPu0v5xC1eob"
# ### Evolution of loss function

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="2lzfhXWi1eob" executionInfo={"status": "ok", "timestamp": 1730893431663, "user_tz": -420, "elapsed": 463, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="23c5547d-3786-4472-d9de-281fc22127c6"
fig, ax = plt.subplots()
ax.semilogy(history.history['loss'], label='Loss')
ax.semilogy(history.history['val_loss'], label='Val Loss')
ax.set_title('Loss')
ax.set_xlabel('Epoch')
ax.grid()
ax.legend()
plt.show()

# %% [markdown] id="XCQnRNKe1eob"
# ### Spot check of phase-space distribution

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="EfpWKzOU1eob" outputId="264281ca-b507-4103-dca8-ececc77c6855"
predictions = model.predict(x_val1)
for i in range(30):
    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].pcolormesh(np.linspace(xmin,xmax,nbins+1), np.linspace(-1.,181.,91), tomodata[n_train+n_test+i,:,:], cmap='rainbow')
    ax[0].set_xlabel("x [mm]")
    ax[0].set_ylabel("angle [deg.]")
    ax[0].set_title("Tomography")
    ax[1].pcolormesh(xedges, yedges, predictions[i,:,:,0].T, cmap='rainbow')
    ax[1].set_xlim((xmin,xmax))
    ax[1].set_ylim((xpmin,xpmax))
    ax[1].set_xlabel("x [mm]")
    ax[1].set_ylabel("x' [mrad]")
    ax[1].set_title("Prediction")
    ax[2].pcolormesh(xedges, yedges, phasespace[n_train+n_test+i,:,:].T, cmap='rainbow')
    ax[2].set_xlim((xmin,xmax))
    ax[2].set_ylim((xpmin,xpmax))
    ax[2].set_xlabel("x [mm]")
    ax[2].set_ylabel("x' [mrad]")
    ax[2].set_title("Ground Truth")
    plt.tight_layout()
    plt.show()

# %% id="5SjYYB5kMBqy"
