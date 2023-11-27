#%%
from readDataset import *
import os
import TestFullModel_specific
import pickle
import numpy as np
import sys
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import pickle
import copy
import seaborn as sns
import pywt
model_name = "patient_specific_chb_DCAE_LSTM_inter_gap_7200"
patient_name = "CHB001"

# matrix, postprocessed_matrix, sens, fpr, seg_results
# seg_results = [filename, start, duration, string_state, true_label, predicted_label]
test_segment = ['/host/d/CHB/CHB001/CHB001_01.edf', 0, 5]
test_data = Segments2Data([test_segment],'chb')



    
#print("Sens", sens_sum/7)
#print("FPR ", fpr_sum/7)


# def SaveAsHeatmap(matrix, path):
#     sns.heatmap(matrix,annot=True, cmap='Blues')
#     plt.xlabel('Predict')
#     plt.ylabel('True')
#     plt.savefig(path)
#     plt.clf()

# %%
