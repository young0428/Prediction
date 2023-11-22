from readDataset import *

train_info_file_path = "/home/CHB/patient_info_chb_train.csv"
test_info_file_path = "/home/CHB/patient_info_chb_test.csv"
edf_file_path = "/home/CHB"

train_interval_set, train_interval_overall = LoadDataset(train_info_file_path)
train_segments_set = {}

patient_name_list = GetPatientName(train_interval_overall)
print(patient_name_list)