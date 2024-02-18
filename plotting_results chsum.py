#%%
from readDataset import *
import os
import pickle
import matplotlib.pyplot as plt
import pyedflib
import datetime

def collectOneFileSegments(segments):
    total = []
    in_one_file = []
    file_name = segments[0][0]
    state = segments[0][3]
    for idx, segment in enumerate(segments):
        if file_name == segment[0] and state == segment[3]:
            in_one_file.append(segment)
            if idx == len(segments) - 1:
                total.append(in_one_file)
        else:
            total.append(in_one_file)
            file_name = segment[0]
            state = segment[3]
            in_one_file = [segment]
            
    return total

def getStartEnd(segments):
    start = segments[0][1]
    end = segments[-1][1] + segments[-1][2]
    duration = end - start
    return start, end, duration

def getEdfStartTime(edf_file_path):
    with pyedflib.EdfReader(edf_file_path) as f:
        edf_start_time = f.getStartdatetime()
        unix_start_time = (edf_start_time - datetime.datetime(1970, 1, 1)).total_seconds()
    return unix_start_time

def getSeizureTime(segments):
    for seg in segments:
        if seg[3] == 'preictal_ontime':
            end = seg[1] + seg[2]
            end_file_name = seg[0]
    unix_start_time = getEdfStartTime(end_file_name)
    seizure_start_time = unix_start_time + end + 120
    return seizure_start_time
        
            
            
    

model_name = "one_ch_dilation_lstm_300sec_random"
sr = 200
channels = ['FP1-F7','T7-P7','FP2-F4','T8-P8','P7-O1','P8-O2']

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
patient_ch_spcific_dict = {}
test_break = False

#%%
patient_info_list = []
for patient_idx, patient_name in enumerate(filtered_interval_dict.keys()) :
        
    train_val_sets = MakeValidationIntervalSet(filtered_interval_dict[patient_name])
    
    patient_acc_sum = 0
    patient_sens_sum = 0
    patient_fpr_sum = 0
    set_num = 0
    
    for idx, set in enumerate(train_val_sets):
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        plt.ylabel("prob")
        plt.axhline(y=0.5, color='r', linestyle='-', label='Threshold Line')
        plt.legend()
        plt.xlabel(f"Time difference from a seizure(min)\n")
        plt.subplot(1,2,2)
        plt.ylabel("prob")
        plt.axhline(y=0.5, color='r', linestyle='-', label='Threshold Line')
        plt.legend()
        plt.xlabel(f"Time difference from a seizure(min)\n")
        test_break = True
        for channel in channels:
            model_name_ch_added = model_name + "_" + channel
            checkpoint_path = f"./Dilation/{model_name_ch_added}/{patient_name}/set{idx+1}/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)
            
            
            if not os.path.exists(f'{checkpoint_dir}/ValResults'):
                continue
            with open(f'{checkpoint_dir}/training_done','r') as f:
                line = f.readline()
                if line == '1':
                    with open(f'{checkpoint_dir}/ValResults', 'rb') as file_pi:
                        # if not os.path.exists(f'{checkpoint_dir}/fixed_done'):
                        #     break
                        result_list = pickle.load(file_pi)
                        matrix = result_list[0]
                        
                        tn = matrix[0][0]
                        fp = matrix[0][1]
                        fn = matrix[1][0]
                        tp = matrix[1][1]

                        acc = (tp + tn) / (tp + tn +fp + fn)
                        sens = tp / (fn + tp)
                        fpr = fp / (tn + fp)

                        seg_results = result_list[4]
                        seg_results = sorted(seg_results, key=lambda x: (x[0], x[1]))
                        splited_segments = collectOneFileSegments(seg_results)
                        seizure_time = getSeizureTime(seg_results)
                        part_num = len(splited_segments)
                        
                        for cnt,continued_segments in enumerate(splited_segments):
                            if not (len(splited_segments) == 2):

                                break
                            
                            edf_name = continued_segments[0][0]
                            state = continued_segments[0][3]
                            true_label = continued_segments[0][4]
                            start, end, duration = getStartEnd(continued_segments)
                            edf_start_time = getEdfStartTime(edf_name)
                            edf_data = Segments2Data([[edf_name, start, duration]],manual_channels=[channel])
                            x = []
                            y = []
                            for seg in continued_segments:
                                x.append(edf_start_time+seg[1]+seg[2])
                                y.append(seg[5])
                            x = np.array(x)
                            x -=seizure_time
                            x /= 60
                            
                           
                            
                            test_break = False
                            plt.subplot(1,2,cnt%2+1)
                            plt.plot(x,y,label=channel)
                            plt.title(f"{state}") 
                            plt.axis([x[0], x[-1], 0, 1])
                            plt.legend()
                            
                                
                            
                            
                            
                        
        
        file_name = f"{patient_name}_set{idx+1}.png"
        plt.suptitle(f"{patient_name}, set {idx+1}")
        plt.tight_layout()
        if test_break == False:
            print(file_name)
            plt.savefig(f"./Dilation/result_fig_chsum/{file_name}")
            
        plt.close()
        
#%%
                        
    
    
#%%
# total_acc_sum = 0
# total_sens_sum = 0
# total_fpr_sum = 0
# channel_sum = []
# for _ in range(len(channels)):
#     channel_sum.append({'acc':0,'sens':0,'fpr':0,'cnt':0})
# for patient_name in patient_ch_spcific_dict.keys():
#     sets = list(patient_ch_spcific_dict[patient_name].keys())
#     patient_acc_sum = 0
#     patient_sens_sum = 0
#     patient_fpr_sum = 0
#     for set in sets:
#         print("--------------------------")
#         print(f"{patient_name} Set {set}")
#         for idx, channel in enumerate(patient_ch_spcific_dict[patient_name][set].keys()):
#             channel_sum[idx]['acc'] += patient_ch_spcific_dict[patient_name][set][channel]['acc']
#             channel_sum[idx]['sens'] += patient_ch_spcific_dict[patient_name][set][channel]['sens']
#             channel_sum[idx]['fpr'] += patient_ch_spcific_dict[patient_name][set][channel]['fpr']
#             channel_sum[idx]['cnt'] += 1
#             print("%6s, Acc : %.2f%% , Sens : %.2f%%, FPR : %.3f"%(channel, 
#                                                                    patient_ch_spcific_dict[patient_name][set][channel]['acc'],
#                                                                    patient_ch_spcific_dict[patient_name][set][channel]['sens'],
#                                                                    patient_ch_spcific_dict[patient_name][set][channel]['fpr'],))
#         print("--------------------------")
#   #%%      
# for idx in range(len(channel)):
#     channel_name = channels[idx]
#     ch_acc = channel_sum[idx]['acc'] / channel_sum[idx]['cnt']
#     ch_sens = channel_sum[idx]['sens'] / channel_sum[idx]['cnt']
#     ch_fpr = channel_sum[idx]['fpr'] / channel_sum[idx]['cnt']
#     print(f"{channel_name} Acc : %.2f%%, Sens : %.2f, FPR : %.3f"%(ch_acc, ch_sens, ch_fpr))

            
    
    
#     pass

        

# %%
