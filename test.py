#%%
from readDataset import *
import os
import pickle
import numpy as np
import sys
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import pickle
import copy
import pywt
from scipy import signal
from vit_tensorflow.mobile_vit import MobileViT
model_name = "patient_specific_chb_DCAE_LSTM_inter_gap_7200"
patient_name = "CHB001"

# matrix, postprocessed_matrix, sens, fpr, seg_results
# seg_results = [filename, start, duration, string_state, true_label, predicted_label]

#%%
test_segment = ['/host/d/CHB/CHB001/CHB001_06.edf', 1000, 8]
test_data = Segments2Data([test_segment],'chb')
test_data = np.squeeze(test_data)
ch1_data = test_data[3]
sampling_rate = 256

plt.figure(figsize=(16,4))
freqs = np.linspace(100,0.1,128) / sampling_rate
scale = pywt.frequency2scale('cgau8',freqs) 
cwtmatr, freq= pywt.cwt(ch1_data, wavelet='cgau8', scales = scale)
cwt_image = np.abs(cwtmatr)

cwt_image = np.expand_dims(cwt_image,axis=-1)

cwt_image = np.expand_dims(cwt_image,axis=0)

test_segment = ['/host/d/CHB/CHB001/CHB001_05.edf', 1000, 8]
test_data = Segments2Data([test_segment],'chb')
test_data = np.squeeze(test_data)
ch1_data = test_data[3]
sampling_rate = 256

plt.figure(figsize=(16,4))
freqs = np.linspace(100,0.1,128) / sampling_rate
scale = pywt.frequency2scale('cgau8',freqs) 
cwtmatr, freq= pywt.cwt(ch1_data, wavelet='cgau8', scales = scale)
cwt_image2 = np.abs(cwtmatr)

cwt_image2 = np.expand_dims(cwt_image2,axis=-1)

cwt_image2 = np.expand_dims(cwt_image2,axis=0)



print(np.shape(cwt_image))

mbvit_xs = MobileViT(
    image_size = (128, 1600),
    # (64 , 800), (32, 400), (16, 200)
    patch_size=(8,25),
    dims = [96, 120, 144],
    channels = [16, 32, 48, 48, 64, 64, 80, 80],
    num_classes = 1000
)

pred = mbvit_xs(cwt_image)
print(np.shape(pred))









    
#print("Sens", sens_sum/7)
#print("FPR ", fpr_sum/7)


# def SaveAsHeatmap(matrix, path):
#     sns.heatmap(matrix,annot=True, cmap='Blues')
#     plt.xlabel('Predict')
#     plt.ylabel('True')
#     plt.savefig(path)
#     plt.clf()

# %%
