#%%
from readDataset import *
import os
import pickle


# model_name = "one_ch_dilation_lstm_300sec_random"
# channels = ['FP1-F7','T7-P7','FP2-F4','T8-P8','P7-O1','P8-O2']
# train_info_file_path = "/host/d/CHB/patient_info_chb_train.csv"
# test_info_file_path = "/host/d/CHB/patient_info_chb_test.csv"
# edf_file_path = "/host/d/CHB"
# data_type = 'chb_one_ch'


train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
edf_file_path = "/host/d/SNU_DATA"
data_type = 'snu_one_ch'

model_name = "snu_one_ch_dilation_lstm_300sec_random"
channels = ["Fp1-AVG", "Fp2-AVG", "T3-AVG", "T4-AVG", "O1-AVG", "O2-AVG"]




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
patient_ch_spcific_dict = {}

#%%
patient_info_list = []
for patient_idx, patient_name in enumerate(filtered_interval_dict.keys()) :
        
    train_val_sets = MakeValidationIntervalSet(filtered_interval_dict[patient_name])
    
    patient_acc_sum = 0
    patient_sens_sum = 0
    patient_fpr_sum = 0
    set_num = 0
    for channel in channels:
        model_name_ch_added = model_name + "_" + channel
        for idx, set in enumerate(train_val_sets):

            checkpoint_path = f"./Dilation/{model_name_ch_added}/{patient_name}/set{idx+1}/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)
            # if idx==1 and patient_name=='CHB014':
            #     print(set['val'])
            if not os.path.exists(f'{checkpoint_dir}/ValResults'):
                #print(idx, "pass")
                continue
            with open(f'{checkpoint_dir}/training_done','r') as f:
                line = f.readline()
                if line == '1':
                    with open(f'{checkpoint_dir}/ValResults', 'rb') as file_pi:
                        if not patient_name in patient_ch_spcific_dict.keys():
                            patient_ch_spcific_dict[patient_name] = {}
                        if not str(idx+1) in patient_ch_spcific_dict[patient_name].keys():
                            patient_ch_spcific_dict[patient_name][str(idx+1)] = {}
                        
                        patient_ch_spcific_dict[patient_name][str(idx+1)][channel] = {}
                        result_list = pickle.load(file_pi)
                        matrix = result_list[0]
                        
                        tn = matrix[0][0]
                        fp = matrix[0][1]
                        fn = matrix[1][0]
                        tp = matrix[1][1]

                        acc = (tp + tn) / (tp + tn +fp + fn)
                        sens = tp / (fn + tp)
                        fpr = fp / (tn + fp)
                        #print("%s set%d, Acc : %.2f%% , Sens : %.2f%%, FPR : %.3f"%(patient_name, idx+1, acc*100, sens*100, fpr))
                        #print(channel)
                        patient_ch_spcific_dict[patient_name][str(idx+1)][channel]['acc'] = acc*100
                        patient_ch_spcific_dict[patient_name][str(idx+1)][channel]['sens'] = sens*100
                        patient_ch_spcific_dict[patient_name][str(idx+1)][channel]['fpr'] = fpr
                        patient_acc_sum += acc
                        patient_sens_sum += sens
                        patient_fpr_sum += fpr
                        set_num += 1
                        
    
    
#%%
total_acc_sum = 0
total_sens_sum = 0
total_fpr_sum = 0
channel_sum = []
for _ in range(len(channels)):
    channel_sum.append({'acc':0,'sens':0,'fpr':0,'cnt':0})
for patient_name in patient_ch_spcific_dict.keys():
    sets = list(patient_ch_spcific_dict[patient_name].keys())
    patient_acc_sum = 0
    patient_sens_sum = 0
    patient_fpr_sum = 0
    for set in sets:
        print("--------------------------")
        print(f"{patient_name} Set {set}")
        for idx, channel in enumerate(patient_ch_spcific_dict[patient_name][set].keys()):
            channel_sum[idx]['acc'] += patient_ch_spcific_dict[patient_name][set][channel]['acc']
            channel_sum[idx]['sens'] += patient_ch_spcific_dict[patient_name][set][channel]['sens']
            channel_sum[idx]['fpr'] += patient_ch_spcific_dict[patient_name][set][channel]['fpr']
            channel_sum[idx]['cnt'] += 1
            print("%6s, Acc : %.2f%% , Sens : %.2f%%, FPR : %.3f"%(channel, 
                                                                   patient_ch_spcific_dict[patient_name][set][channel]['acc'],
                                                                   patient_ch_spcific_dict[patient_name][set][channel]['sens'],
                                                                   patient_ch_spcific_dict[patient_name][set][channel]['fpr'],))
        print("--------------------------")
  #%%      
for idx in range(len(channel)):
    channel_name = channels[idx]
    ch_acc = channel_sum[idx]['acc'] / channel_sum[idx]['cnt']
    ch_sens = channel_sum[idx]['sens'] / channel_sum[idx]['cnt']
    ch_fpr = channel_sum[idx]['fpr'] / channel_sum[idx]['cnt']
    print(f"{channel_name} Acc : %.2f%%, Sens : %.2f, FPR : %.3f"%(ch_acc, ch_sens, ch_fpr))

            
    
    
    pass

        

# %%
