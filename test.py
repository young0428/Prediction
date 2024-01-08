#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import random
import ssqueezepy as ssq


import TestFullModel_specific
from readDataset import *
import AutoEncoder 
from LSTMmodel import LSTMLayer
from sklearn.model_selection import KFold
from ModelGenerator import FullModel_generator
from dilationmodel import *
# %%
s = [['/host/d/CHB/CHB001/CHB001_03.edf',3000,30]]
eeg_data = Segments2Data(s,'chb_one_ch')
eeg_data = np.squeeze(eeg_data)
scale_resolution = 256
sampling_rate = 200

freqs = np.logspace(np.log10(100),np.log10(0.1),scale_resolution) / sampling_rate
#freqs = np.linspace(100,0.1,scale_resolution) / sampling_rate
scale = pywt.frequency2scale('morl',freqs) 
cwtmatr, _= pywt.cwt(eeg_data, wavelet='morl', scales = scale)
# normalize
cwtmatr /= np.abs(cwtmatr).max()
cwt_image = np.abs(cwtmatr)
plt.imshow(cwt_image,cmap='jet',aspect='auto',extent=[-1, 1, 1, scale_resolution])
plt.show()
plt.clf()

ssq_cwtmatr,origin_cwtmatr, *_ = ssq.ssq_cwt(eeg_data,fs=sampling_rate,ssq_freqs=freqs)
ssq_cwtimage = np.abs(ssq_cwtmatr)
ssq_cwtimage /= np.abs(ssq_cwtimage).max()
print(np.shape(ssq_cwtimage))
plt.imshow(ssq_cwtimage,cmap='jet',aspect='auto',extent=[-1, 1, 1, scale_resolution])
plt.show()
plt.clf()

origin_cwtimage = np.abs(origin_cwtmatr)
origin_cwtimage /= np.abs(origin_cwtimage).max()
print(np.shape(origin_cwtimage))
plt.imshow(origin_cwtimage,cmap='jet',aspect='auto',extent=[-1, 1, 1, scale_resolution])
plt.show()
plt.clf()



# %%
