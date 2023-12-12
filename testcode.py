#%%
from readDataset import *
from ModelGenerator import FullModel_generator
import os
import pickle

model_name = "one_ch_chb_dilation_model"

train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
edf_file_path = "/host/d/CHB"

data_type = 'chb_one_ch'


#%%

train_interval_set, train_interval_overall = LoadDataset(train_info_file_path)
train_segments_set = {}
val_segments_set = {}

# %%
channel_filtered_intervals = FilteringByChannel(train_interval_overall, edf_file_path, data_type)
interval_dict_key_patient_name = Interval2NameKeyDict(channel_filtered_intervals)
filtered_interval_dict, ictal_num = FilterValidatePatient(interval_dict_key_patient_name)

channel_filtered_intervals = FilteringByChannel(train_interval_overall, edf_file_path, data_type)
interval_dict_key_patient_name = Interval2NameKeyDict(channel_filtered_intervals)
filtered_interval_dict, ictal_num = FilterValidatePatient(interval_dict_key_patient_name)

total_acc_sum = 0
total_sens_sum = 0
total_fpr_sum = 0
patient_num = 0

#%%
patient_info_list = []
for patient_idx, patient_name in enumerate(filtered_interval_dict.keys()) :
        
    train_val_sets = MakeValidationIntervalSet(filtered_interval_dict[patient_name])

    patient_acc_sum = 0
    patient_sens_sum = 0
    patient_fpr_sum = 0
    set_num = 0
    
    for idx, set in enumerate(train_val_sets):

        checkpoint_path = f"./Dilation/{model_name}/{patient_name}/set{idx+1}/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        if not os.path.exists(checkpoint_dir):
            print(idx, "pass")
            continue
        with open(f'{checkpoint_dir}/training_done','r') as f:
            line = f.readline()
            if line == '1':
                with open(f'{checkpoint_dir}/ValResults', 'rb') as file_pi:
                    result_list = pickle.load(file_pi)
                    matrix = result_list[0]
                    
                    tn = matrix[0][0]
                    fp = matrix[0][1]
                    fn = matrix[1][0]
                    tp = matrix[1][1]

                    acc = (tp + tn) / (tp + tn +fp + fn)
                    sens = tp / (fn + tp)
                    fpr = fp / (tn + fp)
                    print("%s set%d, Acc : %.2f%% , Sens : %.2f%%, FPR : %.3f"%(patient_name, idx+1, acc*100, sens*100, fpr))
                    patient_acc_sum += acc
                    patient_sens_sum += sens
                    patient_fpr_sum += fpr
                    set_num += 1
    
    if set_num > 0:
        patient_avg_acc = patient_acc_sum / set_num
        patient_avg_sens = patient_sens_sum / set_num
        patient_avg_fpr = patient_fpr_sum / set_num
        patient_info_list.append((patient_name, patient_avg_acc*100, patient_avg_sens*100, patient_avg_fpr))
        print('-------------------------------')
        print('Patient %s Avg   Acc : %.2f%%, Sens : %.2f%%, FPR : %.3f'%(patient_name, patient_avg_acc*100, patient_avg_sens*100, patient_avg_fpr))
        print('-------------------------------')

    total_acc_sum += patient_avg_acc
    total_sens_sum += patient_avg_sens
    total_fpr_sum += patient_avg_fpr
    patient_num += 1

total_acc = total_acc_sum / patient_num
total_sens = total_sens_sum / patient_num
total_fpr = total_fpr_sum / patient_num

for info in patient_info_list:
    print('Patient %s Avg   Acc : %.2f%%, Sens : %.2f%%, FPR : %.3f'%(info[0], info[1], info[2], info[3]))

print('########################################################')
print('ToTal       Acc : %.2f%%, Sens : %.2f%%, FPR : %.3f'%(total_acc*100, total_sens*100, total_fpr))

print(patient_num)
        
