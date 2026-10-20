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

# %% [markdown] id="RMnSyHmaPv11"
# # Anomaly detection of a thyratron grid waveform
#
# Copyright (C) 2024 Hirokazu Maesaka, RIKEN SPring-8 Center. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

# %% id="JnZHxKS-sYZv"
import os
import subprocess
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['font.size'] = 16
# %matplotlib inline

# %% [markdown] id="9XbraQFJPv12"
# ## Mount google drive to get data

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 32005, "status": "ok", "timestamp": 1730741803629, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="EdlqgXAyUOXm" outputId="b4b75e10-ac97-4690-b2f2-27401cd2c764"
from google.colab import drive
drive.mount('/content/drive')

# %% id="yNk3x8RvgcL3"
basedir='/content/drive/MyDrive/Colab Notebooks/ISBA24'

# %% [markdown] id="rO-sisYhfff-"
# # Checking (preparing) data files

# %% id="HSsP27kCcCie"
if not os.path.isdir(basedir+'/data'):
  print('Data directory not found, trying to decompress data archive.')
  if not os.path.isfile(f'{basedir}/wfmdata.tar.gz'):
    print('Data archive not found.')
    raise FileNotFoundError
  print(subprocess.check_output(['tar', 'zxvf', basedir+'/wfmdata.tar.gz', '-C', basedir+'/']).decode())
datadir=basedir+'/data'

# %% [markdown] id="XeW-Mad_wFtY"
# # Classification of input data

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 19551, "status": "ok", "timestamp": 1730741823564, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="cJ8DKcfIsYRT" outputId="7a1d4351-e0ad-423e-8061-77ae1a5d9dc2"
thyIdList=[[],[],[]] # new, aged, failed
dfList=[[],[],[]]
thyData=glob.glob(datadir+'/sn*_g1.npz')
for data in thyData:
  dirname=os.path.basename(data)
  serial, hrs=re.findall('sn(\d+)_(\d+)h_', dirname)[0]
  serial=int(serial)
  hrs=int(hrs)

  tIdx=-1
  if hrs<5000: # 1500
    tIdx=0
  elif hrs<20000:
    tIdx=1
  else:
    tIdx=2

  tdf=pd.DataFrame(np.load('%s/sn%d_%dh_g1.npz'%(datadir, serial, hrs))['wfm'])
  thyIdList[tIdx].append([serial, hrs])
  dfList[tIdx].append(tdf)

print('new: %d'%len(thyIdList[0]))
print('aged: %d'%len(thyIdList[1]))
print('failed: %d'%len(thyIdList[2]))

# S/N: 2176 (18100hr) seems to be in good condition

# %% [markdown] id="lMkM7bU6vc3w"
# ## Training data

# %% id="6wY_-CSHvG7J"
tdflist=[]
for i, dfl in enumerate(dfList):
  if i==1: # aged
    continue

  tdflist.append(pd.concat(dfl))
tdf=pd.concat(tdflist)

# %% [markdown] id="1g7IQqORPv14"
# # PCA

# %% colab={"base_uri": "https://localhost:8080/", "height": 80} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1730741823565, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="NWtgW3iHY5aa" outputId="4aab21de-6f21-4d3d-8682-3c76d14c2de4"
pca=PCA(n_components=6)
pca.fit(tdf)

# %% [markdown] id="Rq_92-f0ydE_"
# ## Cumulative contribution ratio

# %% colab={"base_uri": "https://localhost:8080/", "height": 461} executionInfo={"elapsed": 498, "status": "ok", "timestamp": 1730741824050, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="vl93n_TaF3jk" outputId="227681ef-4edd-4d42-f247-7e0c60616585"
cr=pd.DataFrame(np.insert(pca.explained_variance_ratio_,0,0))
plt.plot(cr.cumsum(),'o-')
plt.ylim(0.,1.05)
plt.grid()
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative contribution ratio')
plt.show()

# %% [markdown] id="J_u2ZrcYPv15"
# ## Plot all the PCA outputs

# %% colab={"base_uri": "https://localhost:8080/", "height": 976} id="-zsoYq3KPv15" executionInfo={"status": "ok", "timestamp": 1730741825615, "user_tz": -420, "elapsed": 1581, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="e896bb6c-0143-45f3-9aed-93a35c21b029"
fig=plt.figure(figsize=[8,8])

pcadf=None
mk = ['+','x','.']
for i, dfl in enumerate(dfList):
  for j, df in enumerate(dfl[:10]):
    ptf=pca.transform(df)
    plt.scatter(ptf[:,0], ptf[:,1], marker=mk[i], label='SN: %d (%dhr)'%(thyIdList[i][j][0],thyIdList[i][j][1]))

plt.xlabel('PCA-1')
plt.ylabel('PCA-2')

plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1., 0.5))
plt.show()

# %% [markdown] id="_GRfPAjKyg_Z"
# ## Plot PCA outputs for training data

# %% colab={"base_uri": "https://localhost:8080/", "height": 714} executionInfo={"elapsed": 1283, "status": "ok", "timestamp": 1730741826881, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}, "user_tz": -420} id="FXc_h8SBwD1e" outputId="0c7578bd-ed23-4f12-8215-20cba1876b3b"
fig=plt.figure(figsize=[8,8])

pcadf=None
for i, dfl in enumerate(dfList):
  if i==1: # aged
    continue

  for j, df in enumerate(dfl[:10]):
    ptf=pca.transform(df)
    plt.scatter(ptf[:,0], ptf[:,1], marker='+' if i==0 else '.', label='SN: %d (%dhr)'%(thyIdList[i][j][0],thyIdList[i][j][1]))

    if i==2 and thyIdList[i][j][0]==2010:
      pcadf=pd.DataFrame(ptf[:,:2])

plt.xlabel('PCA-1')
plt.ylabel('PCA-2')

plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1., 0.5))
plt.show()

# %% [markdown] id="QbkrVgMXPv16"
# ## Waveform of each component

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="u4wrzPWvPv16" executionInfo={"status": "ok", "timestamp": 1730741827684, "user_tz": -420, "elapsed": 834, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="e095cbed-2aa3-43f0-e8cc-2b760237c3f8"
for i in range(pca.components_.shape[0]):
    fig, ax = plt.subplots()
    ax.plot(pca.components_[i,:])
    ax.grid()
    ax.set_title(f'Component {i+1}')
    plt.show()

# %% [markdown] id="oScMUc6UPv16"
# ## Linear regression

# %% id="UjR6Vw26Pv16"
from sklearn.linear_model import LinearRegression

# %% colab={"base_uri": "https://localhost:8080/"} id="F9HKclgQPv16" executionInfo={"status": "ok", "timestamp": 1730741828106, "user_tz": -420, "elapsed": 429, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="cccd764f-e630-4879-bb94-1850106f7946"
# Select new and failed tubes
xtrainlist = []
ytrainlist = []
xtestlist = []
ytestlist = []
for i, dfl in enumerate(dfList):
    for j, df in enumerate(dfl):
        ptf=pca.transform(df)
        if i==1:
            xtestlist.append(ptf)
            ytestlist.append(np.zeros(ptf.shape[0]))
        else:
            xtrainlist.append(ptf)
            ytrainlist.append(np.ones(ptf.shape[0])*(i//2))
xtrain = np.vstack(xtrainlist)
ytrain = np.hstack(ytrainlist)
xtest = np.vstack(xtestlist)
ytest = np.hstack(ytestlist)
print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)

# %% id="StTQ2zeoPv17"
# Fit
clf = LinearRegression().fit(xtrain,ytrain)

# %% colab={"base_uri": "https://localhost:8080/", "height": 464} id="zW4icUfxPv17" executionInfo={"status": "ok", "timestamp": 1730741828748, "user_tz": -420, "elapsed": 646, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="b297688c-4b11-41d2-ea1c-fed80494b804"
fig, ax = plt.subplots()
ax.hist(clf.predict(xtrain[ytrain==0]), 100, [-0.5,1.5], alpha=0.8, label='train new')
ax.hist(clf.predict(xtrain[ytrain==1]), 100, [-0.5,1.5], alpha=0.8, label='train failed')
ax.hist(clf.predict(xtest), 100, [-0.5,1.5], alpha=0.8, label='test aged')
ax.grid()
ax.legend()
ax.set_title('Linear regression')
plt.show()

# %% [markdown] id="Aos251ssPv17"
# ## Support vector machine (SVM)

# %% id="5lTfzpmXPv17"
from sklearn import svm

# %% colab={"base_uri": "https://localhost:8080/", "height": 80} id="TfdcdoN2Pv17" executionInfo={"status": "ok", "timestamp": 1730741828748, "user_tz": -420, "elapsed": 23, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="70c836d4-3a45-450b-d80a-01711ed7667f"
# Support vector machine regressor
clf = svm.SVR()
clf.fit(xtrain, ytrain)

# %% colab={"base_uri": "https://localhost:8080/", "height": 464} id="Pki-5wzePv17" executionInfo={"status": "ok", "timestamp": 1730741829729, "user_tz": -420, "elapsed": 1003, "user": {"displayName": "Hirokazu Maesaka", "userId": "11909388552187572989"}} outputId="428e6394-509e-4dd1-960f-53a1361cf305"
fig, ax = plt.subplots()
ax.hist(clf.predict(xtrain[ytrain==0]), 100, [-0.5,1.5], alpha=0.8, label='train new')
ax.hist(clf.predict(xtrain[ytrain==1]), 100, [-0.5,1.5], alpha=0.8, label='train failed')
ax.hist(clf.predict(xtest), 100, [-0.5,1.5], alpha=0.8, label='test aged')
ax.grid()
ax.legend()
ax.set_title('SVM regression')
plt.show()

# %% id="3XCGYB19MRlR"
