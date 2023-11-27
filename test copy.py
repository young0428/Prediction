#%%
from readDataset import *
import os
import TestFullModel_specific
import pickle

model_name = "patient_specific_chb_DCAE_LSTM_inter_gap_7200"
patient_name = "CHB001"

# matrix, postprocessed_matrix, sens, fpr, seg_results
# seg_results = [filename, start, duration, string_state, true_label, predicted_label]
sens_sum = 0
fpr_sum = 0
right = 0
wrong = 0
int_right = 0
int_wrong = 0

with open(f'/host/c/Users/hy105/Desktop/Prediction/LSTM/paper_base_128ch_categorical_chb_binary/MatrixHistory', 'rb') as file_pi:
    result_list = pickle.load(file_pi)
    print(result_list[0][0][1]/(result_list[0][0][1]+result_list[0][0][0]) * (3600 / 5))
        

    



# def SaveAsHeatmap(matrix, path):
#     sns.heatmap(matrix,annot=True, cmap='Blues')
#     plt.xlabel('Predict')
#     plt.ylabel('True')
#     plt.savefig(path)
#     plt.clf()

# %%
