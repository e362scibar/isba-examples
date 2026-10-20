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

# %% [markdown] id="gZ2i1pK0wL7R"
# # Regression of emittance and Twiss parameters from quadrupole scan data
# - Use Gaussian beam
#

# %% [markdown] id="wok7GUis_iWw"
#
# Copyright (C) 2024 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
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
from scipy.stats import gaussian_kde, gamma, multivariate_normal
import math
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown] id="wbkna0PQwL7X"
#
# ## Parameters

# %% id="EQI1TRVWwL7X"
N = 10000   # number of data samples
M = 10000   # number of particles
L = 5.0     # drift length [m]
f0inv = 1.0 # 1/f_0 (reciprocal of minimum focal length of the Q)
f1inv = 0.0 # 1/f_1 (reciprocal of maximum focal length of the Q)
NQ = 21     # number of Q settings

# %% [markdown] id="y8fijaKewL7X"
# ## Generate emittance and Twiss parameters randomly

# %% [markdown] id="9WIikJs-wL7Y"
# ### Distribution of emittance
# - Gamma distribution
# - Natural emittance ~ 5 $\mu$ m

# %% colab={"base_uri": "https://localhost:8080/", "height": 470} id="g-myY06GwL7Y" executionInfo={"status": "ok", "timestamp": 1730891737639, "user_tz": -420, "elapsed": 1559, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="73837cd1-faaa-4ef2-866c-c8a54ce12959"
gammarv = gamma(a=5, scale=1.0e-6)
emittance = gammarv.rvs(size=N)
fig, ax = plt.subplots()
y,x,h = ax.hist(emittance*1.0e6, bins=100)
ax.set_xlabel("emittance [$\mu$m rad]")
plt.show()
print(emittance.min(), emittance.mean())

# %% [markdown] id="04Zc9-ATwL7Z"
# ### Distribution of beta
# - Gamma distribution
# - beta ~ 10 m

# %% colab={"base_uri": "https://localhost:8080/", "height": 466} id="Rw_OfPDSwL7Z" executionInfo={"status": "ok", "timestamp": 1730891738360, "user_tz": -420, "elapsed": 724, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="39b4b4e4-a6bc-4ee0-b9bf-16a71b009d97"
gammarv = gamma(a=10, scale=1.0)
beta = gammarv.rvs(size=N)
fig, ax = plt.subplots()
y,x,h = ax.hist(beta, bins=100)
ax.set_xlabel("beta [m]")
plt.show()
print(beta.min(), beta.mean())

# %% [markdown] id="xdXX2o2RwL7Z"
# ### Distribution of alpha
# - uniform distribution within the boundary
# - The boundary condition is the existance of a beam waist on the screen.
# - $\beta(1/L - 1/f_0) < \alpha < \beta(1/L - 1/f_1)$

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} id="XwyesbPHwL7a" executionInfo={"status": "ok", "timestamp": 1730891738360, "user_tz": -420, "elapsed": 15, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="1c6dccd6-43a6-4d51-e2e4-4672402d6daa"
a0, a1 = beta*(1.0/L - f0inv), beta*(1.0/L - f1inv)
urand = np.random.uniform(size=N)
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

# %% id="5j1HHuqswL7a"
particles = np.empty((N,M,2))
mean = np.zeros(2)
for i in range(N):
    particles[i,:,:] = multivariate_normal.rvs(mean=mean, cov=cov[i,:,:], size=M)

# %% [markdown] id="7AfmX3uwwL7a"
# ### Check distribution

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="zovVPZRWwL7b" executionInfo={"status": "ok", "timestamp": 1730891769749, "user_tz": -420, "elapsed": 15893, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="391d6b0d-a7f8-42ef-8084-842c58521fda"
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

# %% colab={"base_uri": "https://localhost:8080/"} id="yWa5EQ9qwL7b" executionInfo={"status": "ok", "timestamp": 1730891826964, "user_tz": -420, "elapsed": 57220, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="68363e67-8359-4438-92ff-de3fba11d44e"
Nbin = 60
Xmin = -30.0e-3
Xmax =  30.0e-3
qscan = np.empty((N, NQ, Nbin))
for i in range(NQ):
    print(i)
    p1 = tmat[i,:,:] @ np.swapaxes(particles, 1, 2)
    for j in range(N):
        h,xe = np.histogram(p1[j,0,:], Nbin, (Xmin,Xmax))
        qscan[j,i,:] = h


# %% [markdown] id="qNNiEYOKwL7b"
# ### Check Q-scan data

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="ndSbqGIawL7c" executionInfo={"status": "ok", "timestamp": 1730891828495, "user_tz": -420, "elapsed": 1539, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="90a7d08a-722f-4e8e-bd0d-ab1446eea45a"
for i in range(10):
    fig, ax = plt.subplots(figsize=(3,2))
    df = (f0inv-f1inv)/(NQ-1)
    ax.pcolormesh(np.linspace(Xmin,Xmax,Nbin+1)*1.e3, np.linspace(f1inv-df*0.5,f0inv+df*0.5,NQ+1), qscan[i,:,:], cmap='rainbow')
    #ax.set_xlim((-200,200))
    #ax.set_ylim((-20,20))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Q strength [a.u.]")
    plt.show()

# %% [markdown] id="vAvpyUsrwL7c"
# ## Prepare data for CNN

# %% [markdown] id="06uIa0ZuwL7c"
# ### Split data into training, test, and validation

# %% id="J3H2YVy4wL7c"
n_train, n_test, n_val = 6000, 2000, 2000
y_scale = np.array([1.0e5, 0.05, 0.01])
y_data = np.array([emittance, alpha, beta]).T * y_scale
x_train, y_train = qscan[:n_train,:,:], y_data[:n_train,:]
x_test, y_test = qscan[n_train:n_train+n_test,:,:], y_data[n_train:n_train+n_test,:]
x_val, y_val = qscan[-n_val:,:,:], y_data[-n_val:,:]

# %% [markdown] id="jUhPQXanwL7c"
# ### Input image dimensions

# %% id="IiREwYZ3wL7c"
img_rows, img_cols = Nbin, NQ
input_shape = (img_rows, img_cols, 1)

# %% [markdown] id="f2TwkOuuwL7c"
# ### Reshape and scale input data

# %% id="Ek9wEXU4wL7c"
x_train1 = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype(np.float32)/M
x_test1 = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype(np.float32)/M
x_val1 = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1).astype(np.float32)/M

# %% [markdown] id="dWNGaF7vwL7d"
# ## Training of CNN

# %% [markdown] id="wvrXLgYfwL7d"
# ### Import modules

# %% id="enPYDwPcwL7d"
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 397} id="zw7bmqjPwL7h" executionInfo={"status": "ok", "timestamp": 1730891832413, "user_tz": -420, "elapsed": 23, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="c83169cb-7431-410a-d0c3-f6dd2909721d"
model.summary()

# %% [markdown] id="0WfvoVXGwL7i"
# ### Fit the model to data (Training)

# %% colab={"base_uri": "https://localhost:8080/"} id="Y06_NgZIwL7i" executionInfo={"status": "ok", "timestamp": 1730891853131, "user_tz": -420, "elapsed": 20738, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="b9a06e2b-137e-4281-8c90-711f18ffaef6"
history = model.fit(x_train1, y_train, batch_size=2*batch_size, epochs=epochs, verbose=1, validation_data=(x_test1, y_test))
score = model.evaluate(x_test1, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# %% [markdown] id="7Y9-yxbAwL7i"
# ### Evolution of loss function

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="861vDJMTwL7i" executionInfo={"status": "ok", "timestamp": 1730891853131, "user_tz": -420, "elapsed": 15, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="b11da858-837a-468b-d1ce-8d6e829e2c19"
fig, ax = plt.subplots()
ax.semilogy(history.history['loss'], label='Loss')
ax.semilogy(history.history['val_loss'], label='Val Loss')
ax.set_title('Loss')
ax.set_xlabel('Epoch')
ax.grid()
ax.legend()
plt.show()

# %% [markdown] id="zCdes0VzwL7i"
# ## Check results

# %% [markdown] id="s9CW29LswL7i"
# ### Spot check

# %% colab={"base_uri": "https://localhost:8080/"} id="I34dCi12wL7i" executionInfo={"status": "ok", "timestamp": 1730891853917, "user_tz": -420, "elapsed": 800, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="21a45ce3-f82b-402d-ea56-b1be00633eb6"
predictions = model.predict(x_val1)
print('                emittance,      alpha,       beta')
for i in range(10):
    print('prediction  ', predictions[i]/y_scale)
    print('ground truth', y_val[i]/y_scale)
    print('error       ', (y_val[i]-predictions[i])/y_scale)
    print()

# %% [markdown] id="HhxpRoauwL7j"
# ### Compare phase-space distribution

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="yg_nOdE9wL7j" executionInfo={"status": "ok", "timestamp": 1730891891020, "user_tz": -420, "elapsed": 37114, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="14f862d1-a4c8-4bba-8df0-f6ed94988740"
for i in range(10):
    data = particles[n_train+n_test+i,:,:]
    eps, a, b = predictions[i]/y_scale
    g = (1+a**2)/b
    cov = eps * np.array([[b, -a],[-a, g]])
    pred = multivariate_normal.rvs(mean=np.zeros(2), cov=cov, size=M)
    z = gaussian_kde(data.T)(data.T)
    idx = z.argsort()
    z2 = z[idx]
    z = gaussian_kde(pred.T)(pred.T)
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
    ax[1].scatter(pred[idx2,0]*1.0e3, pred[idx2,1]*1.0e3, c=z3, s=1, cmap='rainbow', edgecolor='none')
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

# %% colab={"base_uri": "https://localhost:8080/", "height": 969} id="FQHc6lEYwL7j" executionInfo={"status": "ok", "timestamp": 1730891891698, "user_tz": -420, "elapsed": 698, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="31d5bbd3-9c37-4aa9-e901-9fed2253d968"
pred = predictions/y_scale
pred_emittance = pred[:,0]
pred_alpha = pred[:,1]
pred_beta = pred[:,2]
truth = y_val/y_scale
truth_emittance = truth[:,0]
truth_alpha = truth[:,1]
truth_beta = truth[:,2]
# Emittance
xy = np.vstack([truth_emittance, pred_emittance])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
z2 = z[idx]
fig, ax = plt.subplots(figsize=(3,3))
ax.scatter(truth_emittance[idx]*1.0e6, pred_emittance[idx]*1.0e6, c=z2, s=1, cmap='rainbow', edgecolor='none')
ax.grid()
ax.set_axisbelow(True)
ax.set_xlabel("Truth [$\mu$m rad]")
ax.set_ylabel("Prediction [$\mu$m rad]")
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


# %% id="N6WLrO6VwL7j"
