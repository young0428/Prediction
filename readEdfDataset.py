import pyedflib
# 데이터 정리 
name = None
f = None
def SegmentToData(segments):
    # segment[0] = 'filename', segment[1] = 'start', segment[2] = 'duration'
    for segment in segments:
        if not name == segment[0]:
            name = segment[0]
            if not f == None:
                f.close()
            f = pyedflib.EdfReader(segment[0])
            freq = f.getSampleFrequencies
        
        f.read