import numpy as np
list = [ [[1],[2],[3]],[[1],[2],[3]],[[1],[2],[3]]  ]
print(np.shape(list))
a = np.array(list)
b = np.split(a,3,axis=-2)
print(np.shape(b))