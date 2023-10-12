import numpy as np
import random
import operator
def GetBatchIndexes(data_len, batch_num):
    batch_size = data_len / batch_num
    idx_list = list(range(data_len))
    random.shuffle(idx_list)

    batch_idx_mask = []
    for i in range(batch_num):
        batch_idx_mask.append(sorted( idx_list[int(batch_size*i) : int(batch_size*(i+1))] ))

    return batch_idx_mask






