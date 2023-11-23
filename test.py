#%%
from readDataset import *

train_info_file_path = "/host/d/SNU_DATA/patient_info_snu_train.csv"
test_info_file_path = "/host/d/SNU_DATA/patient_info_snu_test.csv"
edf_file_path = "/host/d/SNU_DATA"

train_interval_set, train_interval_overall = LoadDataset(train_info_file_path)
train_segments_set = {}

# %%
channel_filtered_intervals = FilteringByChannel(train_interval_overall, edf_file_path, 'snu')
patient_name_list = GetPatientName(channel_filtered_intervals)
interval_dict_key_patient_name = Interval2NameKeyDict(channel_filtered_intervals)
filtered_interval_dict, ictal_num = FilterValidatePatient(interval_dict_key_patient_name)
for patient_name in filtered_interval_dict.keys() :
    set = MakeValidationIntervalSet(filtered_interval_dict[patient_name])
    

# %%
