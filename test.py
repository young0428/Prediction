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
matrix_avg = np.array([[0,0],[0,0]], dtype=int)
for idx in range(7):
    with open(f'./LSTM/{model_name}/{patient_name}/set{idx+1}/ValResults', 'rb') as file_pi:
        result_list = pickle.load(file_pi)
        # sens_sum += result_list[1][1][1]/(result_list[1][1][1] + result_list[1][1][0])
        # fpr_sum += result_list[1][0][1]/(result_list[1][0][1] + result_list[1][0][0])
        
        # matrix = np.array(result_list[0], dtype=int)
        # matrix_avg += matrix

        # for it in result_list[4]:
        #     if it[3] == 'preictal_early':
        #         if it[4] == it[5]:
        #             right += 1
        #         else:
        #             wrong+=1
        #     else:
        #         if it[3] == 'interictal' or it[3] == 'postictal':
        #             if it[4] == it[5]:
        #                 int_right += 1
        #             else:
        #                 int_wrong+=1

print(right/(right+wrong))
print(int_right/(int_right+int_wrong))
print(matrix_avg)

print("Sens, ",matrix_avg[1][1] / (matrix_avg[1][0]+matrix_avg[1][1]))
print("ACC, ",(matrix_avg[1][1] + matrix_avg[0][0]) / (matrix_avg[1][0]+matrix_avg[1][1]+matrix_avg[0][0]+matrix_avg[0][1]))
print("FPR, ",matrix_avg[0][1] / (matrix_avg[0][0]+matrix_avg[0][1]) * 3600/5)

sns.heatmap(matrix_avg,annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predict')
plt.ylabel('True')
plt.savefig(f'./LSTM/{model_name}/{patient_name}/avg_matrix')
plt.clf()

    
#print("Sens", sens_sum/7)
#print("FPR ", fpr_sum/7)


# def SaveAsHeatmap(matrix, path):
#     sns.heatmap(matrix,annot=True, cmap='Blues')
#     plt.xlabel('Predict')
#     plt.ylabel('True')
#     plt.savefig(path)
#     plt.clf()

# %%
