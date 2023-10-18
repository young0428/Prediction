import numpy as np
import random
import operator
def GetBatchIndexes(data_len, batch_num):
    batch_size = data_len / batch_num
    idx_list = list(range(data_len))
    # batch_seg_size = batch_size / 20
    # idx_list = [ list(range(int(i*batch_seg_size), int((i+1)*batch_seg_size))) for i in range(batch_num*20) ]
    #random.shuffle(idx_list)

    batch_idx_mask = []
    for i in range(batch_num):
        #batch_idx_mask.append(np.concatenate(idx_list[int(20*i) : int(20*(i+1))]))
        batch_idx_mask.append(sorted( idx_list[int(batch_size*i) : int(batch_size*(i+1))] ))
    return batch_idx_mask








