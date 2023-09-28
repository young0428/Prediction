import pyedflib
import numpy as np
# 데이터 정리 

def SegmentToData(segments):
    # segment[0] = 'filename', segment[1] = 'start', segment[2] = 'duration'
    signal_for_all_segments = []
    name = None
    f = None
    for segment in segments:
        if not name == segment[0]:
            name = segment[0]
            if not f == None:
                f.close()
            f = pyedflib.EdfReader(segment[0])
            print("file opened")
        freq = f.getSampleFrequencies()
        labels = f.getSignalLabels()
        chn_num = len(labels)
        seg = []
        x = np.linspace(0, 10,int(segment[2]*freq[0]) )
        x_upsample = np.linspace(0,10,int(256*segment[2]))

        for i in range(chn_num-1):
            skip = False
            for j in range(i):
                if labels[i] == labels[j]:
                    skip = True
            if skip:
                continue
            signal = f.readSignal(i,segment[1],int(freq[i]*segment[2]))
            # 256 Hz이하일 경우 256Hz로 interpolation을 이용한 upsampling
            if not freq[i] == 256:
                signal = np.interp(x_upsample,x, signal)
            
            seg.append(signal)
            
        
        signal_for_all_segments.append(seg)
    
    if hasattr(f,'close'):
         f.close()

    return signal_for_all_segments


####    test code    ####
test = [ ["E:/SNUH_START_END/patient_3/patient_3.edf",0,2],["E:/SNUH_START_END/patient_3/patient_3.edf",2,2] ]
a = SegmentToData(test)
