import pyedflib
import numpy as np
import pandas as pd
import copy
import PreProcessing
import pywt
from scipy.signal import resample
# 데이터 정리 
global state_list
state_list = ['ictal', 'preictal_late', 'preictal_early', 'preictal_ontime', 'postictal','interictal']

def MakePath(filename, data_path):
    return data_path+'/'+(filename.split('_'))[0]+'/'+filename+'.edf'

def LoadDataset(filename):
    df = pd.read_csv(filename)
    columns = ['name','start','end']
    columns2 = ['name','start','end','state']
    interval_dict = {}
    for state in state_list:
        condition = df['state']  ==  state
        df_state = df[condition]
        interval_dict[state] = df_state[columns].values.tolist()

    whole_interval = df[columns2].values.tolist()

    return interval_dict, whole_interval

    
def get_first_name_like_layer(model,name):  
    for layer in model.layers:
        if name in layer.name:
            return layer

# %%
def GetPatientName(intervals):
    patient_name_list = []
    for interval in intervals:
        if not (interval[0].split('_'))[0] in patient_name_list:
            patient_name_list.append((interval[0].split('_'))[0])
    return patient_name_list

def IntervalFilteringByName(intervals, patient_name):
    interval_for_name = []
    for interval in intervals:
        # CHB001_01 -> (CHB001,01), (CHB001,01)[0] = CHB001
        if (interval[0].split('_'))[0] == patient_name:
            interval_for_name.append(interval)
    return interval_for_name

#interval = [name, start, end, state]
# interval과 state를 주면 환자 이름별로 interval 모아줌
def Interval2NameKeyDict(origin_intervals):
    patient_name_list = GetPatientName(origin_intervals)
    interval_dict_key_patient_name = {}
    for patient_name in patient_name_list:
        interval_dict_key_patient_name[patient_name] = IntervalFilteringByName(origin_intervals, patient_name)

    return interval_dict_key_patient_name

# ictal이 2번 이상인 환자만 뽑아서 dictionary return
def FilterValidatePatient(interval_dict):
    ictal_count = {}
    validate_patient_dict = {}
    for patient_name in interval_dict.keys():
        ictal_cnt = 0
        for idx, interval in enumerate(interval_dict[patient_name]):
            # ictal이 파일 사이에 걸쳐 있을 경우, state가 ictal인 interval이 2번 연속으로 나올 경우
            # 뒤의 ictal은 count 안함
            if interval[3] == 'ictal':
                if idx-1 > 0:
                    if interval_dict[patient_name][idx-1][3]  ==  'ictal':
                        continue
                ictal_cnt += 1
        ictal_count[patient_name] = ictal_cnt
        if ictal_cnt >= 2 :
            validate_patient_dict[patient_name] = interval_dict[patient_name]
            ictal_count[patient_name] = ictal_cnt
    return validate_patient_dict, ictal_count

def FilteringByChannel(intervals, edf_path, type):
    channels_snu = ['Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'Fp2-AVG', 'F4-AVG', 
                    'C4-AVG', 'P4-AVG', 'F7-AVG', 'T1-AVG', 'T3-AVG', 'T5-AVG', 
                    'O1-AVG', 'F8-AVG', 'T2-AVG', 'T4-AVG', 'T6-AVG', 'O2-AVG', 'Fz-AVG', 'Cz-AVG', 'Pz-AVG']
    
    channels_chb = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3',
                    'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                    'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8',
                    'P8-O2', 'FZ-CZ', 'CZ-PZ']
    channels_snu_one = ['Fp1-AVG']
    channels_chb_one = ['F8-T8']

    if type == 'snu':
        channels = channels_snu
    elif type == 'chb':
        channels = channels_chb
    elif type == 'snu_one_ch':
        channels = channels_snu_one
    elif type == 'chb_one_ch':
        channels = channels_chb_one
    copied_intervals = copy.deepcopy(intervals)
    
    mask = np.ones(len(copied_intervals),dtype=bool)
    del_cnt = 0
    for idx,interval in enumerate(intervals):
        with pyedflib.EdfReader(MakePath(interval[0], edf_path)) as f:
            labels = f.getSignalLabels()
            if not all([channel in labels for channel in channels]):
                del copied_intervals[idx-del_cnt]
                del_cnt+=1

    return copied_intervals

def MakeValidationIntervalSet(patient_specific_intervals, least_preictal = 300, least_interictal=1800):
    true_state = ['preictal_ontime', 'preictal_late', 'preictal_early', 'ictal']
    false_state =['interictal']
    start_idx = -1
    end_idx = -1
    start_time = -1
    end_time = -1
    train_val_set = []
    t3_period = least_interictal
    preictal_period = 0
    least_preictal_period = least_preictal
    least_interictal_period = least_interictal
    set_cnt = 0
    for idx, interval in enumerate(patient_specific_intervals):
        if interval[3] in true_state:
            if start_idx  ==  -1:
                start_idx = idx

            if (start_idx != -1) and (interval[3] == 'ictal'):
                if idx+1 < len(patient_specific_intervals):
                    if patient_specific_intervals[idx+1][3] == 'ictal' : continue
                set_cnt += 1
                end_idx = idx
                val_idx_list = []
                state2find = 'interictal'
                direction = 'backward'
                done_flag = False
                remain_period = t3_period
                intervals_copied = copy.deepcopy(patient_specific_intervals)
                train_val_dict = {'train':[], 'val':[]}
                interval_idx = start_idx
                pre_inter_gap = 7200

                while True:
                    interval_idx = FindStateIntervalIdx(intervals_copied, interval_idx, direction, state2find)
                    if done_flag :
                        done_flag = False
                        interval_idx = -1
                        
                    if interval_idx  ==  -1:
                        if (state2find == 'interictal') and  (direction == 'backward'):
                            if remain_period <= 0:
                                #state2find = 'postictal'
                                # remain_period = t3_period
                                # pre_inter_gap = 7200
                                val_idx_list += list(range(start_idx, end_idx+1))
                                break
                            direction = 'forward'
                            interval_idx = end_idx
                            continue
                        if (state2find == 'interictal') and (direction == 'forward'):
                            # state2find='postictal'
                            # remain_period = t3_period
                            # pre_inter_gap = 7200
                            # continue
                            val_idx_list += list(range(start_idx, end_idx+1))
                            break
                        # if state2find  ==  'postictal':
                        #     val_idx_list += list(range(start_idx, end_idx+1))
                        #     break
                    interictal_period = intervals_copied[interval_idx][2] - intervals_copied[interval_idx][1]
                    if remain_period - interictal_period > 0 : 
                        val_idx_list.append(interval_idx)
                        remain_period -= interictal_period
                        done_flag = False
                        continue
                    if remain_period - interictal_period  ==  0:
                        val_idx_list.append(interval_idx)
                        remain_period -= interictal_period
                        done_flag = True
                        continue
                    if remain_period - interictal_period < 0:
                        temp = copy.deepcopy(intervals_copied[interval_idx])
                        if direction == 'backward':
                            temp[1] = intervals_copied[interval_idx][2] - remain_period
                            train_val_dict['val'].append(temp)
                            intervals_copied[interval_idx][2] = temp[1]

                        elif direction == 'forward':
                            temp[2] = intervals_copied[interval_idx][1] + remain_period
                            train_val_dict['val'].append(temp)
                            intervals_copied[interval_idx][1] = temp[2]
                        done_flag = True
                        remain_period -= interictal_period

                        continue
                
                start_idx = -1
                

                intervals_copied = np.array(intervals_copied)
                train_mask = np.ones(len(intervals_copied), dtype=bool)
                train_mask[val_idx_list] = False
                train_val_dict['train'] = intervals_copied[train_mask].tolist()

                inter_sum = 0
                preictal_sum = 0
                for interval in train_val_dict['train']:
                    if interval[3] == 'interictal':
                        inter_sum += (int(interval[2])-int(interval[1]))
                        continue
                    if interval[3] == 'preictal_ontime':
                        preictal_sum += (int(interval[2])-int(interval[1]))
                        continue

                if inter_sum < least_interictal_period or preictal_sum < least_preictal_period:
                    #print(f"train set {set_cnt+1} has not enough period")
                    continue

                val_mask = np.zeros(len(intervals_copied), dtype=bool)
                val_mask[val_idx_list] = True
                train_val_dict['val'] += intervals_copied[val_mask].tolist()
                
                inter_sum = 0
                preictal_sum = 0
                for interval in train_val_dict['val']:
                    if interval[3] == 'interictal':
                        inter_sum += (int(interval[2])-int(interval[1]))
                        continue
                    if interval[3] == 'preictal_ontime':
                        preictal_sum += (int(interval[2])-int(interval[1]))
                        continue

                if inter_sum < least_interictal_period or preictal_sum < least_preictal_period:
                    #print(f"val set {set_cnt+1} has not enough period")
                    continue


                train_val_set.append(train_val_dict)
                

    return train_val_set

def FindStateIntervalIdx(patient_specific_intervals, idx, direction, state):
    while True:
        if direction == 'forward':
            idx += 1
            if idx >= len(patient_specific_intervals) : return -1
        if direction == 'backward':
            idx -= 1
            if idx < 0 : return -1
        if patient_specific_intervals[idx][3] == state: return idx


# state = ['ictal', 'preictal_late', 'preictal_early', 'preictal_ontime', 'postictal','interictal']
# output = [name, start, window_size]
def Interval2Segments(interval_list, data_path, window_size, sliding_size):
    segments_list = []
    for interval in interval_list:
        start = interval[1]
        end = interval[2]
        if end - start < window_size:
            continue
        segment_num = int(((end-start-window_size)/sliding_size))+1
        for i in range(segment_num):
            segments_list.append([MakePath(interval[0],data_path), start, window_size])
            start += sliding_size

    return segments_list





def Segments2Data(segments, type='snu', manual_channels=None):
    # segment[0] = 'filename', segment[1] = 'start', segment[2] = 'duration'
    channels_snu = ['Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'Fp2-AVG', 'F4-AVG', 'C4-AVG', 'P4-AVG', 'F7-AVG', 'T1-AVG', 'T3-AVG', 'T5-AVG', 'O1-AVG', 'F8-AVG', 'T2-AVG', 'T4-AVG', 'T6-AVG', 'O2-AVG', 'Fz-AVG', 'Cz-AVG', 'Pz-AVG']
    channels_chb = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3',
                    'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4',
                    'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8',
                    'P8-O2', 'FZ-CZ', 'CZ-PZ']
    channels_snu_one = ['Fp1-Avg']
    channels_chb_one = ['FP2-F8']
    if type == 'snu':
        channels = channels_snu
    elif type == 'chb':
        channels = channels_chb
    elif type == 'snu_one_ch':
        channels = channels_snu_one
    elif type == 'chb_one_ch':
        channels = channels_chb_one 
    
    if manual_channels != None:
        channels = manual_channels


    signal_for_all_segments = []
    name = None
    read_end = 0
    f = None
    cnt = 0
    for idx, segment in enumerate(segments):
        with pyedflib.EdfReader(segment[0]) as f:
            skip_start = False   # 연속된 시간이면 한번에 읽기 위해 파일 읽는 시작 시간은 그대로 두고 끝 시간만 갱신함

            freq = f.getSampleFrequencies()
            labels = f.getSignalLabels()
            if not all([channel in labels for channel in channels]):
                f.close()
                continue
            
            if not skip_start:
                interval_sets = [] # 연속된 구간이면 한번에 읽고 구간 정해진거에 따라 나누기 위해 구간 저장
                read_start = float(segment[1])
                read_end = 0
            # 최근 세그먼트의 start+window_size 값보다 read_end 값이 작으면 (읽는 끝값) read_end 값 갱신
            
            read_end = float(segment[1]) + float(segment[2])
            interval_sets.append([float(segment[1])-read_start, float(segment[1])+float(segment[2])-read_start ])

        
            skip_start = False

            chn_num = len(channels)

            # UpSampling rate
            target_sampling_rate = 200

            seg = []
            for i in range(len(interval_sets)):
                seg.append([])

            
            for channel in channels:
                ch_idx = labels.index(channel)
                signal = f.readSignal(ch_idx,int(freq[ch_idx]*read_start),int(freq[ch_idx]*float(segment[2])))
                #128가 아닐 경우 256Hz로 interpolation을 이용한 upsampling
                if not freq[ch_idx] == target_sampling_rate:
                    signal = resample(signal, int(float(segment[2]) * target_sampling_rate ))
                    
                
                # for j in range(len(interval_sets)):
                #     seg[j].append( list(signal[int(interval_sets[j][0] * target_sampling_rate) : int(interval_sets[j][1] * target_sampling_rate) ]) )
                for j in range(len(interval_sets)):
                    seg[j].append( list(signal) )
            f.close()
        
            for s in seg:    
                signal_for_all_segments.append(s)

            
            skip_start = False

    return np.array(signal_for_all_segments)

def updateDataSet(type_1_len, type_2_len, type_3_len, portion, batch_size):
    type_1_devided_by_portion = type_1_len/portion[0]
    type_2_devided_by_portion = type_2_len/portion[1]
    type_3_devided_by_portion = type_3_len/portion[2]

    if type_1_len == 0 : type_1_devided_by_portion = np.inf
    if type_2_len == 0 : type_2_devided_by_portion = np.inf
    if type_3_len == 0 : type_3_devided_by_portion = np.inf

    n = int(min(type_1_devided_by_portion, type_2_devided_by_portion, type_3_devided_by_portion))

    if type_1_len == 0 : type_1_sample_num = 0
    else : type_1_sample_num = int(n*portion[0])
    if type_2_len == 0 : type_2_sample_num = 0
    else : type_2_sample_num = int(n*portion[0])
    if type_3_len == 0 : type_3_sample_num = 0
    else : type_3_sample_num = int(n*portion[0])
    

    if (type_1_sample_num+type_2_sample_num+type_3_sample_num) < batch_size :
        type_1_sample_num += int(batch_size)
        type_3_sample_num += int(batch_size)
    
    type_1_sample_num = min(type_1_len, type_1_sample_num)
    type_3_sample_num = min(type_3_len, type_3_sample_num)
    batch_num = int((type_1_sample_num+type_2_sample_num+type_3_sample_num)/batch_size)

    # Sampling mask 생성
    
    type_1_sampling_mask = np.array(sorted(np.random.choice(type_1_len, type_1_sample_num, replace=False)))
    type_2_sampling_mask = np.array(sorted(np.random.choice(type_2_len, type_2_sample_num, replace=False)))
    type_3_sampling_mask = np.array(sorted(np.random.choice(type_3_len, type_3_sample_num, replace=False)))
    
    type_1_batch_indexes = PreProcessing.GetBatchIndexes(type_1_sample_num, batch_num, 0)
    type_2_batch_indexes = PreProcessing.GetBatchIndexes(type_2_sample_num, batch_num, 0)
    type_3_batch_indexes = PreProcessing.GetBatchIndexes(type_3_sample_num, batch_num, 0)

    return [type_1_sampling_mask, type_1_batch_indexes, type_2_sampling_mask,  type_2_batch_indexes, type_3_sampling_mask, type_3_batch_indexes], batch_num

def IntervalList2Dict(intervals):
    interval_dict = {}
    for interval in intervals:
        if not interval[3] in interval_dict.keys():
            interval_dict[interval[3]] = []
        
        interval_dict[interval[3]].append(interval)

    return interval_dict


####    test code    ####
#test = [ ["D:/SNU_DATA/SNU003/SNU003.edf",0,2],["D:/SNU_DATA/SNU003/SNU003.edf",2,2],["D:/SNU_DATA/SNU003/SNU003.edf",3,2],
#         ["D:/SNU_DATA/SNU003/SNU003.edf",5,2],["D:/SNU_DATA/SNU003/SNU003.edf",5,2]]
#a = Segments2Data(test)
# data = LoadDataset('C:/Users/hy105/Desktop/Prediction/patient_info.csv')
# segments = Interval2Segments(data['ictal'],2,1)
#print(segments[:100])
