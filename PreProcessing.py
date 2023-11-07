import numpy as np
import random
import operator
import scipy
def GetBatchIndexes(data_len, batch_num):
    batch_size = data_len / batch_num
    mul = 50
    #idx_list = list(range(data_len))
    batch_seg_size = batch_size / mul
    idx_list = [ list(range(int(i*batch_seg_size), int((i+1)*batch_seg_size))) for i in range(int(batch_num*mul)) ]
    random.shuffle(idx_list)

    batch_idx_mask = []
    for i in range(batch_num):
        batch_idx_mask.append(np.asarray(np.concatenate(sorted(idx_list[int(mul*i) : int(mul*(i+1))])), dtype=int))
        #batch_idx_mask.append(sorted( idx_list[int(batch_size*i) : int(batch_size*(i+1))] ))
    return batch_idx_mask

def BandPassfiltering(signal, bandwidth, sr):
    sos = scipy.signal.butter(3, [bandwidth[0], bandwidth[1]], 'band', fs=sr, output='sos')
    filtered_signal = scipy.signal.sosfilt(sos, signal)

    return filtered_signal

def FilteringSegments(segments):
    y_batch = []
    for batch in segments:
        fft_seg = []
        for one_ch_signal in batch:
            fft_seg.append(BandPassfiltering(signal=one_ch_signal, bandwidth=[0.1, 50], sr=128))
        y_batch.append(fft_seg)
    y_batch = np.array(y_batch)
    return y_batch

def AbsFFT(signal):
    filtered = np.fft.fft(signal) / len(signal)
    filtered_abs = abs(filtered)

    return filtered_abs









