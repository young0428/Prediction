#%%
import os
import pyedflib
import datetime
import re

import numpy as np
def get_chb_summary_info(summary_path: str):
    with open(summary_path, 'r') as f:
        contents = f.read().split("\n\n")
    
    seizure_info_list = []

    for info in contents:
        try:
            info = info.split("\n")
            file_name_idx = None
            number_of_seizures_idx = None
            
            for idx, line in enumerate(info):
                if ('Name' in line) or ('name' in line):
                    file_name_idx = idx
                
                if ('Number' in line) or ('number' in line):
                    number_of_seizures_idx = idx

            # exception point
            # if "number of seizure" not in info: Exception
            num_of_seizure = int(list(info[number_of_seizures_idx].split(": "))[-1])
            
            if num_of_seizure>0:
                filename = list(info[file_name_idx].split(': '))[-1]
                seizure_file = {'name': filename,
                                'seizure': []}

                for i in range(num_of_seizure):
                    seizure_start_idx = number_of_seizures_idx + 2*i + 1
                    seizure_end_idx = number_of_seizures_idx + 2*i + 2
                    seizure_start = int(info[seizure_start_idx].split(": ")[-1].rstrip(' seconds'))
                    seizure_end = int(info[seizure_end_idx].split(": ")[-1].rstrip(' seconds'))
                    seizure_file['seizure'].append([seizure_start, seizure_end])
                
                seizure_info_list.append(seizure_file)     
        
        except:
            pass
                
    return seizure_info_list

def get_patient_file_list(patient_dir):
    file_list = []
    for root, dirs, files in os.walk(patient_dir):
        for file in files:
            if file.endswith(".edf"):
                file_list.append(patient_dir+"/"+file)
    return file_list
#%%
# date to unix timestamp
def date2unix(time):
    return (time - datetime.datetime(1970, 1, 1)).total_seconds()

# get start time, end time of a file
def get_file_start_end_time(file_path):
    with pyedflib.EdfReader(file_path) as f:
        unix_start_time = int(date2unix(f.getStartdatetime()))
    with pyedflib.EdfReader(file_path) as f:
        unix_end_time = int(date2unix(f.getStartdatetime()) + f.getFileDuration())
    return unix_start_time, unix_end_time

# get start time of first file and end time of last file as timestamp
def get_start_end_stamp(file_list):
    start_file = file_list[0]
    end_file = file_list[-1]
    unix_start_time, _= get_file_start_end_time(start_file)
    _, unix_end_time = get_file_start_end_time(end_file)
    
    return [int(unix_start_time), int(unix_end_time)]

def merge_intervals_with_gap(intervals, gap_sec):
    merged_intervals = []
    current_interval = None

    for interval in intervals:
        if current_interval is None:
            current_interval = interval[:]
        else:
            # Check if the gap between the current interval and the new one is within the specified limit
            if interval[0] - current_interval[1] <= gap_sec:
                # Merge the intervals
                current_interval[1] = interval[1]
            else:
                # Gap is exceeded, add the current interval and start a new one
                merged_intervals.append(current_interval[0:2])
                current_interval = interval[:]

    # Add the last interval
    if current_interval is not None:
        merged_intervals.append(current_interval)

    # Format the output with individual intervals before merging
    final_result = []
    for merged_interval in merged_intervals:
        # Extract individual intervals before merging
        individual_intervals = [interval for interval in intervals if merged_interval[0] <= interval[0] <= merged_interval[1]]
        final_result.append([merged_interval, individual_intervals])

    return final_result


# chb01_01 -> CHB001_01
def name_formatting(filename):
    return filename[:3].upper() + filename[3:5].zfill(3) + filename[5:]
    
def get_chb_interval_info():
    chb_dir = "/host/d/CHB"
    patient_list = [ "CHB%03d"%(i+1) for i in range(24) ]
    patient_list.remove("CHB017")
    patient_file_list = {}
    patient_start_end = {}
    total_time_flag = {}
    patient_total_validation_duration = {}
    enable_period_flag = {}
    patient_seizure_info = {}
    patient_enable_interval_info = {}
    interval_info = {}
    SOP = 30
    SPH = 2
    interictal_gap = 10800 # sec
    early_gap = 3600
    ontime_gap = 60*(SOP+SPH)
    late_gap = 60*SPH
    num_to_state_dict={0:'postictal', 1:'ictal', 2:'preictal_early', 3:'preictal_ontime', 4:'preictal_late', 5:'interictal'}

    for patient_name in patient_list:
        
        patient_file_list[patient_name] = sorted(get_patient_file_list(f"{chb_dir}/{patient_name}"))
        patient_seizure_info[patient_name] = get_chb_summary_info(f"{chb_dir}/{patient_name}/{patient_name}_summary.txt")  
        patient_start_end[patient_name] = get_start_end_stamp(patient_file_list[patient_name])
        
        
    for patient_name in patient_list:
        patient_start_time = patient_start_end[patient_name][0]
        patient_end_time = patient_start_end[patient_name][1]
        patient_duration = patient_end_time - patient_start_time
        seizure_time_flag = np.array([5]*patient_duration)
        
        enable_period_flag[patient_name] = np.array([0]*patient_duration)  
        interval_info[patient_name] = []
        
        
        # check interval flag for real exisitied file
        for file in patient_file_list[patient_name]:
            file_start, file_end = get_file_start_end_time(file)
            file_start_pos = file_start - patient_start_time
            file_end_pos = file_end - patient_start_time
            enable_period_flag[patient_name][file_start_pos:file_end_pos] = 1
            interval_info[patient_name].append([file_start_pos, file_end_pos, file])
            
            
        seizure_time_set = []
        patient_enable_interval_info[patient_name] = merge_intervals_with_gap(interval_info[patient_name], 300)
        # calculate total duration for patient
        patient_total_validation_duration[patient_name] = 0
        for final_interval in patient_enable_interval_info[patient_name]:
            patient_total_validation_duration[patient_name] += final_interval[0][1] - final_interval[0][0]
            

        for name_seizure_set in patient_seizure_info[patient_name]:
            file_path = chb_dir + "/" + patient_name + "/" + name_formatting(name_seizure_set['name'])
            
            for seizure in name_seizure_set['seizure']:
                seizure_start_time = seizure[0]
                seizure_end_time = seizure[1]
                file_start_time,_ = get_file_start_end_time(file_path)
                file_start_pos = file_start_time - patient_start_time
                ictal_start_pos = file_start_pos + seizure_start_time
                ictal_end_pos = file_start_pos + seizure_end_time
                seizure_time_set.append([ictal_start_pos, ictal_end_pos])
        
        # edit time_flag_list following seizure info    
        for i in range(len(seizure_time_set)):
            seizure_start_time = seizure_time_set[i][0]
            seizure_end_time = seizure_time_set[i][1]
            # seizure 시작 전 3시간 0으로 초기화
            # if not seizure_start_time - interictal_gap < 0:
            #     seizure_time_flag[seizure_start_time - interictal_gap :seizure_start_time] = 0
            # else:
            #     seizure_time_flag[0:seizure_start_time] = 0
            # seizure 끝난 후 3시간 0으로 초기화
            if not seizure_end_time + interictal_gap >= patient_duration:
                seizure_time_flag[seizure_end_time : seizure_end_time + interictal_gap] = 0
            else:
                seizure_time_flag[seizure_end_time : patient_duration] = 0

            ## 밑으로 갈수록 우선순위 높음
            ## 덮어씌워짐
        for i in range(len(seizure_time_set)):
            seizure_start_time = seizure_time_set[i][0]
            seizure_end_time = seizure_time_set[i][1]
            # preictal_early 부분 2로 만듦
            if not seizure_start_time - early_gap < 0 :
                seizure_time_flag[seizure_start_time - early_gap : seizure_start_time] = 2
            else:
                seizure_time_flag[0 : seizure_start_time] = 2
        for i in range(len(seizure_time_set)):
            seizure_start_time = seizure_time_set[i][0]
            seizure_end_time = seizure_time_set[i][1]
            # preictal_ontime 부분 3으로 만듦
            if not seizure_start_time - ontime_gap < 0:
                seizure_time_flag[seizure_start_time - ontime_gap : seizure_start_time] = 3
            else:
                seizure_time_flag[0 : seizure_start_time] = 3
        for i in range(len(seizure_time_set)):
            seizure_start_time = seizure_time_set[i][0]
            seizure_end_time = seizure_time_set[i][1]
            # preictal_late 부분 4으로 만듦
            if not seizure_start_time - late_gap < 0 :
                seizure_time_flag[seizure_start_time - late_gap : seizure_start_time] = 4
            else:
                seizure_time_flag[0 : seizure_start_time] = 4

        for i in range(len(seizure_time_set)):
            seizure_start_time = seizure_time_set[i][0]
            seizure_end_time = seizure_time_set[i][1]
            seizure_time_flag[seizure_start_time:seizure_end_time] = 1
        
        total_time_flag[patient_name] = seizure_time_flag
        
    return total_time_flag, patient_enable_interval_info, patient_total_validation_duration, enable_period_flag


    
    
            

            
            
    
            
         

    
    
        
    
        
        
    
    #%%
    
        
    
    
        