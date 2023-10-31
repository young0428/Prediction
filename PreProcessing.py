import numpy as np
import random
import operator
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









