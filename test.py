import numpy as np
import tensorflow as tf
a = 4
b = 4
c = 4
y = [[[1,1,1],[2,2,2],[3,3,3],[4,4,4]]] + [[[5,5,5],[6,6,6],[7,7,7],[8,8,8]]] + [[[9,9,9],[10,10,10],[11,11,11],[12,12,12]]] + [[[13,13,13],[14,14,14],[15,15,15],[16,16,16]]]
tmp2 = []
for a1 in range(6):
    tmp1 = []
    for a2 in range(4):
        tmp1.append([a1*4+a2+1]*3)
    tmp2.append(tmp1)

y = np.array(tmp2)
y.shape = (1,6,4,3)

patches = tf.image.extract_patches(
    images = y,
    sizes = [1, 2,2,1],
    strides=[1,2,2,1],
    rates=[1,1,1,1],
    padding='VALID'
)
print(y[0,:,:,0])
reshaped = tf.reshape(patches,shape=(1,4,6,3))
print(reshaped[0,:,0,0])
reshaped = tf.transpose(reshaped, [0,2,1,3])
print(reshaped[0,:,0,0])
print(patches)