import pyedflib
import numpy as np
import pandas as pd
import sys
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
# output = [name, start, window_size]
def Interval2Segments(interval_list, data_path, window_size, sliding_size):
    segments_list = []
    for interval in interval_list:
        start = interval[1]
        end = interval[2]
        segment_num = int(((end-start-window_size)/sliding_size))+1
        for i in range(segment_num):
            segments_list.append([data_path+'/'+interval[0]+'/'+interval[0]+'.edf', start, window_size])
            start += sliding_size

    return segments_list


def Segments2Data(segments):
    # segment[0] = 'filename', segment[1] = 'start', segment[2] = 'duration'
    channels = ['Fp1-AVG', 'F3-AVG', 'C3-AVG', 'P3-AVG', 'Fp2-AVG', 'F4-AVG', 'C4-AVG', 'P4-AVG', 'F7-AVG', 'T1-AVG', 'T3-AVG', 'T5-AVG', 'O1-AVG', 'F8-AVG', 'T2-AVG', 'T4-AVG', 'T6-AVG', 'O2-AVG', 'Fz-AVG', 'Cz-AVG', 'Pz-AVG']
    signal_for_all_segments = []
    name = None
    read_end = 0
    f = None
    cnt = 0
    for idx, segment in enumerate(segments):
        if (idx+1) % (int(len(segments)/20)) == 0 :
            cnt+=1
            print(f"Data Loading {idx+1} / {len(segments)} ({cnt*5}%)... ")
            

        if not name == segment[0]:
            name = segment[0]
            if not f == None:
                f.close()
            f = pyedflib.EdfReader(segment[0])
            skip_start = False   # 연속된 시간이면 한번에 읽기 위해 파일 읽는 시작 시간은 그대로 두고 끝 시간만 갱신함


        if not skip_start:
            interval_sets = [] # 연속된 구간이면 한번에 읽고 구간 정해진거에 따라 나누기 위해 구간 저장
            read_start = segment[1]
            read_end = 0
        # 최근 세그먼트의 start+window_size 값보다 read_end 값이 작으면 (읽는 끝값) read_end 값 갱신
        if read_end < segment[1] + segment[2]:
            read_end = segment[1] + segment[2]
        interval_sets.append([segment[1]-read_start, segment[1]+segment[2]-read_start ])

        if not idx+1 >= len(segments) :
            # 파일이름이 같고, 다음 세그먼트의 시작시간이 더 크면서 현재 세그먼트의 시작시간 + window_size가 다음 세그먼트의 시작이랑 이어질 때
            if (name == segments[idx+1][0]) and (segment[1] <= segments[idx+1][1]) and (segment[1] + segment[2] >= segments[idx+1][1]) :
                skip_start = True
                continue
        skip_start = False
                
        freq = f.getSampleFrequencies()
        labels = f.getSignalLabels()
        chn_num = len(channels)

        # UpSampling을 위해 x 값 생성
        x = np.linspace(0, 10,int((read_end-read_start)*freq[0]) )
        x_upsample = np.linspace(0,10,int(256*(read_end-read_start)))

        seg = []
        for i in range(len(interval_sets)):
            seg.append([])
        try :
            for channel in channels:
                ch_idx = labels.index(channel)
                edf_signal = f.readSignal(ch_idx,read_start,int(freq[ch_idx]*(read_end-read_start)))
                
                # 256 Hz이하일 경우 256Hz로 interpolation을 이용한 upsampling
                if not freq[ch_idx] == 256:
                    signal = np.interp(x_upsample,x, edf_signal)
                
                for j in range(len(interval_sets)):
                    seg[j].append( list(signal[interval_sets[j][0] * 256 : interval_sets[j][1] * 256 ]) )
        
            for s in seg:    
                signal_for_all_segments.append(s)

        except Exception as e:
            print("Freq : %d"%freq[ch_idx])
            print("labels")
            print(labels)
            print("Read start : %d  Read end : %d"%(read_start,read_end))
            print("minus : %d sample num : %d"%(read_end-read_start, (read_end-read_start)*freq[ch_idx]))
            print(np.shape(x))
            print(np.shape(edf_signal))
            print(np.shape(x_upsample))
            print(np.shape(seg[j]))
            print(f.getFileDuration())
            print(name)
            print(segment)
            print(e)
            sys.exit()
        skip_start = False
    if hasattr(f,'close'):
         f.close()

    return np.array(signal_for_all_segments)


####    test code    ####
#test = [ ["D:/SNU_DATA/SNU003/SNU003.edf",0,2],["D:/SNU_DATA/SNU003/SNU003.edf",2,2],["D:/SNU_DATA/SNU003/SNU003.edf",3,2],
#         ["D:/SNU_DATA/SNU003/SNU003.edf",5,2],["D:/SNU_DATA/SNU003/SNU003.edf",5,2]]
#a = Segments2Data(test)
# data = LoadDataset('C:/Users/hy105/Desktop/Prediction/patient_info.csv')
# segments = Interval2Segments(data['ictal'],2,1)
#print(segments[:100])
