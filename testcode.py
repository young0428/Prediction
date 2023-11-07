import numpy as np
iden_mat = np.eye(3)
y_batch = np.concatenate( ( np.ones(3)*0, 
                                   (np.ones(3))*1, 
                                   (np.ones(3))*2), dtype='int32')
y_batch = np.asarray(y_batch, dtype='int32')
y_batch = iden_mat[y_batch]
print(y_batch)