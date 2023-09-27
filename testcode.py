import pyedflib
edf_path = ""

ictal_state = []


ictal_segments = []
window_size = 2
current_sec = 0
for ictal_interval in ictal_state:
    current_sec = ictal_interval[1] # [0] == name, [1] == start, [2] == end, [3] == state
    while(1):
        if current_sec + window_size > ictal_interval[2]:
            break
        # [filename, start, duration] 형태로 저장
        ictal_segments.append( [edf_path+'/'+ictal_interval[0]+'.edf', current_sec, window_size ] )
        current_sec += window_size




