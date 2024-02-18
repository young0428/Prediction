import numpy as np
import random
import pywt
import ssqueezepy as ssq
import scipy
import os


def GetBatchIndexes(data_len, batch_num, mult=20):
    batch_size = data_len / batch_num
    idx_list = list(range(data_len))
    random.shuffle(idx_list)

    batch_idx_mask = []
    for i in range(batch_num):
        ix_list = sorted(idx_list[int(batch_size*i) : int(batch_size*(i+1))])
        if len(ix_list) != 0 :
            batch_idx_mask.append(np.asarray((ix_list), dtype=int))
        else:
            batch_idx_mask.append([])
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
            fft_seg.append(BandPassfiltering(signal=one_ch_signal, bandwidth=[0.1, 50], sr=200))
        y_batch.append(fft_seg)
    y_batch = np.array(y_batch)
    return y_batch

def AbsFFT(signal):
    filtered = np.fft.fft(signal) / len(signal)
    filtered_abs = abs(filtered)

    return filtered_abs

# Raw EEG batch를 받아서 CWT를 수행한 후 (Batch_size, scale_resolution, window_size * freq) return
def SegmentsCWT(segments, sampling_rate, scale_resolution = 128):
    cwt_result = []
    for segment in segments:
        eeg_data = np.squeeze(segment)
        freqs = np.logspace(np.log10(100),np.log10(0.1),scale_resolution) / sampling_rate
        #freqs = np.linspace(100,0.1,scale_resolution) / sampling_rate
        scale = pywt.frequency2scale('cgau8',freqs) 
        cwtmatr, *_= pywt.cwt(eeg_data, scales = scale, wavelet='cgau8')
        #cwtmatr, *_= ssq.cwt(eeg_data, scales = scale, wavelet='gmw')
        #cwtmatr = cwtmatr.cpu()
        # normalize
        cwt_image = np.abs(cwtmatr)
        cwt_image /= np.abs(cwt_image).max()
        

        cwt_result.append(cwt_image)

    return cwt_result


def SegmentsSSQCWT(segments, sampling_rate, scale_resolution = 128):
    cwt_result = []
    for segment in segments:
        eeg_data = np.squeeze(segment)
        freqs = np.logspace(np.log10(100),np.log10(0.1),scale_resolution) / sampling_rate
        #freqs = np.linspace(100,0.1,scale_resolution) / sampling_rate
        scale = pywt.frequency2scale('morl',freqs) 
        cwtmatr, *_= ssq.ssq_cwt(eeg_data, fs = sampling_rate, wavelet='morlet', scales = scale)
        cwtmatr = cwtmatr.cpu()
        # normalize
        cwt_image = np.abs(cwtmatr)
        cwt_image /= np.abs(cwt_image).max()
        

        cwt_result.append(cwt_image)

    return cwt_result









