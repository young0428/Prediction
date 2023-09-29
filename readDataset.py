import pyedflib
import numpy as np
import pandas as pd
# 데이터 정리 
global state_list
state_list = ['ictal', 'preictal_late', 'preictal_early', 'preictal_ontime', 'postictal','interictal']
def LoadDataset(filename):
    df = pd.read_csv(filename)
    columns = ['name','start','end']
    interval_dict = {}
    for state in state_list:
        condition = df['state'] == state
        df_state = df[condition]
        interval_dict[state] = df_state[columns].values.tolist()

    return interval_dict


# state = ['ictal', 'preictal_late', 'preictal_early', 'preictal_ontime', 'postictal','interictal']
def Interval2Segments(interval_list, window_size, sliding_size):
    segments_list = []
    for interval in interval_list:
        start = interval[1]
        end = interval[2]
        segment_num = int(((end-start-window_size)/sliding_size))+1
        for i in range(segment_num):
            segments_list.append([interval[0], start, window_size])
            start += sliding_size

    return segments_list


def Segments2Data(segments):
    # segment[0] = 'filename', segment[1] = 'start', segment[2] = 'duration'
    signal_for_all_segments = []
    name = None
    read_end = 0
    f = None
    for idx, segment in enumerate(segments):
        if not name == segment[0]:
            name = segment[0]
            if not f == None:
                f.close()
            f = pyedflib.EdfReader(segment[0])
            skip_start = False   # 연속된 시간이면 한번에 읽기 위해 파일 읽는 시작 시간은 그대로 두고 끝 시간만 갱신함
            print("file opened")
        if not skip_start:
            interval_sets = [] # 연속된 구간이면 한번에 읽고 구간 정해진거에 따라 나누기 위해 구간 저장
            read_start = segment[1]
        if read_end < segment[1] + segment[2]:
            read_end = segment[1] + segment[2]
        interval_sets.append([segment[1]-read_start, segment[1]+segment[2]-read_start ])

        print(interval_sets)
        if not idx+1 >= len(segments):
            if segment[1] + segment[2] >= segments[idx+1][1] :
                skip_start = True
                continue   
                
        freq = f.getSampleFrequencies()
        labels = f.getSignalLabels()
        chn_num = len(labels)


        x = np.linspace(0, 10,int((read_end-read_start)*freq[0]) )
        x_upsample = np.linspace(0,10,int(256*(read_end-read_start)))

        print(read_start)
        print(int(freq[0]*(read_end-read_start)))
        seg = []
        for i in range(len(interval_sets)):
            seg.append([])

        for i in range(chn_num-1):
            chn_skip = False
            for j in range(i):
                if labels[i] == labels[j]:
                    chn_skip = True
            if chn_skip:
                continue
            signal = f.readSignal(i,read_start,int(freq[i]*(read_end-read_start)))
            
            # 256 Hz이하일 경우 256Hz로 interpolation을 이용한 upsampling
            if not freq[i] == 256:
                signal = np.interp(x_upsample,x, signal)
            
            for j in range(len(interval_sets)):
                seg[j].append( list(signal[interval_sets[j][0] * 256 : interval_sets[j][1] * 256 ]) )
                
            
        
                
                
        
        for s in seg:
            signal_for_all_segments.append(s)
        skip_start = False
    if hasattr(f,'close'):
         f.close()

    return signal_for_all_segments


####    test code    ####
test = [ ["E:/SNUH_START_END/patient_3/patient_3.edf",0,2],["E:/SNUH_START_END/patient_3/patient_3.edf",2,2],["E:/SNUH_START_END/patient_3/patient_3.edf",3,1],
        ["E:/SNUH_START_END/patient_3/patient_3.edf",5,2],["E:/SNUH_START_END/patient_3/patient_3.edf",5,1]]
a = Segments2Data(test)
#data = LoadDataset('C:/Users/dudgb/Desktop/Prediction/patient_info.csv')
#segments = Interval2Segments(data['ictal'],2,1)
#print(segments[:100])
