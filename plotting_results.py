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

sr = 200


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
    if test_break:
        break
    for channel in channels:
        model_name_ch_added = model_name + "_" + channel
        if test_break:
            break
        for idx, set in enumerate(train_val_sets):
            checkpoint_path = f"./Dilation/{model_name_ch_added}/{patient_name}/set{idx+1}/cp.ckpt"
            checkpoint_dir = os.path.dirname(checkpoint_path)
            # if idx==0 and patient_name=='CHB002':
            #     print(set['val'])
            if not os.path.exists(f'{checkpoint_dir}/ValResults'):
                #print(idx, "pass")
                continue
            with open(f'{checkpoint_dir}/training_done','r') as f:
                line = f.readline()
                if line == '1':
                    with open(f'{checkpoint_dir}/ValResults', 'rb') as file_pi:
                        # if not os.path.exists(f'{checkpoint_dir}/fixed_done'):
                        #     break
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
                        # print("%s set%d, Acc : %.2f%% , Sens : %.2f%%, FPR : %.3f"%(patient_name, idx+1, acc*100, sens*100, fpr))
                        # print(channel)
                        # patient_ch_spcific_dict[patient_name][str(idx+1)][channel]['acc'] = acc*100
                        # patient_ch_spcific_dict[patient_name][str(idx+1)][channel]['sens'] = sens*100
                        # patient_ch_spcific_dict[patient_name][str(idx+1)][channel]['fpr'] = fpr
                        
                        # if not (patient_name == 'CHB014' and channel == 'FP1-F7' and idx+1 == 1):
                        #     pass
                        
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
                            
                            
                            #ax2 = plt.subplot(2,1,2)
                            if cnt % 2 == 0:
                                plt.close()
                                plt.clf()
                                plt.figure(figsize=(12, 6))
                                
                                
                            
                            
                            plt.subplot(1,2,cnt%2+1)
                            plt.plot(x,y,label=channel)
                            plt.title(f"{state}")
                            plt.ylabel("prob")
                            plt.legend()
                            period_acc = "%.2f%%"%((matrix[true_label][true_label] / (matrix[true_label][0] + matrix[true_label][1])) * 100)
                            # for ss in continued_segments:
                            #     print(ss)
                            plt.xlabel(f"Time difference from a seizure(min)\nPeriod Acc : {period_acc}")
                            
                            plt.axhline(y=0.5, color='r', linestyle='-', label='Threshold Line')
                            plt.axis([x[0], x[-1], 0, 1])
                            
                            if cnt%2 == 1:
                                file_name = f"{patient_name}_{idx+1}_{channel.replace('-','')}.png"
                                s = "Acc : %.2f%% , Sens : %.2f%%, FPR : %.3f"%(acc*100, sens*100, fpr)
                                plt.suptitle(f"{patient_name} set {idx+1}, {channel}\n{s}")
                                plt.tight_layout()
                                fig_path = f"./Dilation/{model_name}/plot"
                                if not os.path.exists(fig_path):
                                    os.makedirs(fig_path)
                                plt.savefig(f"{fig_path}/{file_name}")
                                print(file_name)
                                
                                plt.close()
                            
                            
                            
                            # plt.subplot(2,1,1,sharex=ax2)
                            # edf_data_x = np.linspace(x[0]-5, x[-1], len(edf_data[0][0]))
                            # plt.plot(edf_data_x, edf_data[0][0])
                            # plt.ylabel("Amp")
                            # plt.xticks(visible=False)
                            
                            
                                
                        #test_break = True
                        
                            
                        #quit()
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
