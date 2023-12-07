import os
import natsort
import pyedflib
import pandas as pd
import csv
import numpy as np

data_path = "E:/SNUH_START_END"

file_list = natsort.natsorted(os.listdir(data_path)) # 이름순으로 순서 정렬
patient_folder_list = []

for file in file_list:
	file_path = data_path + '/' + file
	if os.path.isdir(file_path):
		patient_folder_list.append([file_path,file])
# [0] : 폴더 경로 	[1] : 폴더 내의 .txt파일 이름 ex) patient_1
total_seizure_info_list = []

#{'name':환자 이름, 'ictal':[ [start,end], [start,end], ... ] } 형태의 환자별 딕셔너리로 만든 후 리스트에 추가
#{'EndTime' : Value } 는 EDF에서 파일 읽어서 끝나는 시간 계산
for patient in patient_folder_list:
	temp_set = {'name': patient[1]}
	with open(patient[0]+'/'+patient[1]+'.txt','r') as f:
		temp_list = []
		for line in f:
			time = line.strip()
			seizure_time_set = (time.split(','))	# 쉼표로 시작 끝시간 구분
			temp_list.append([int(seizure_time_set[0]), int(seizure_time_set[1])])	
		temp_set['ictal'] = temp_list
	
	with pyedflib.EdfReader(patient[0]+'/'+patient[1]+'.edf') as f:
		duration = f.getFileDuration()
		temp_set['endtime'] = duration
	total_seizure_info_list.append(temp_set)


SOP = 30
SPH = 2
interictal_gap = 10800 # sec
early_gap = 3600
ontime_gap = 60*(SOP+SPH)
late_gap = 60*SPH
num_to_state_dict={0:'None', 1:'ictal', 2:'preictal_early', 3:'preictal_ontime', 4:'preictal_late', 5:'interictal'}

for info in total_seizure_info_list:
	info['ictal'].sort(key= (lambda x:x[0]) )	# 시작 시간 순서대로 정렬
	seizure_time_set = info['ictal']
	seizure_time_flag = np.array([5]*info['endtime'])
	# 1 == seizure, 4 == preictal_late, 3 == preictal_ontime, 2 == preictal_early, 5==interictal
	
	interictal_list = []
	preictal_1hour_list = []
	preictal_list = []
	preictal_late_list = []
	post_ictal_list = []
	for i in range(len(seizure_time_set)):
		seizure_start_time = seizure_time_set[i][0]
		seizure_end_time = seizure_time_set[i][1]
		# seizure 시작 전 3시간 0으로 초기화
		if not seizure_start_time - interictal_gap < 0:
			seizure_time_flag[seizure_start_time - interictal_gap :seizure_start_time] = 0
		else:
			seizure_time_flag[0:seizure_start_time] = 0
		# seizure 끝난 후 3시간 0으로 초기화
		if not seizure_end_time + interictal_gap >= info['endtime']:
			seizure_time_flag[seizure_end_time : seizure_end_time + interictal_gap] = 0
		else:
			seizure_time_flag[seizure_end_time : info['endtime']] = 0

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

		
	state = -1
	for i in range(6):
		info[num_to_state_dict[i]] = None

	for i in range(len(seizure_time_flag)):
		if seizure_time_flag[i] != state or i+1 == info['endtime']:
			if state != -1:
				if info[num_to_state_dict[state]] == None:
					info[num_to_state_dict[state]] = []
				info[num_to_state_dict[state]].append([start_sec, i])
				start_sec = i
				state = seizure_time_flag[i]
			else:
				start_sec = 0
				state = seizure_time_flag[0]

patient_segments_list = []
for patient in total_seizure_info_list:
	patient_number = int((patient['name'].split('_'))[1])
	patient_name_snu = "SNU%03d"%patient_number
	dict_keys = list(patient.keys())
	for i in range(len(dict_keys)):
		if dict_keys[i] == 'name' or dict_keys[i] == 'endtime' or dict_keys[i] == "None":
			continue
		time_list = patient[dict_keys[i]]
		if not time_list==None :
			for time in time_list:
				patient_segments_list.append([patient_name_snu, time[0], time[1], dict_keys[i]])

df = pd.DataFrame(patient_segments_list,columns=['name','start','end','state'])
df.to_csv('./patient_info.csv',index=False)












