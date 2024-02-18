#%%
import keras
from vit_tensorflow.vit import ViT
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def BoradAttentionViT(one_ch_inputs):
    head_unit_num = 128
    v = ViT(
        image_shape = (one_ch_inputs.shape[-3], one_ch_inputs.shape[-2], one_ch_inputs.shape[-1]),
        patch_shape = (32, 200),
        projection_dim = 64,
        num_heads = 8,
        transformer_units = [64],
        mlp_head_units = [head_unit_num],
        num_layers = 8,
    )

    vit_outputs, Q, K, V = v.create_vit_layer(inputs = one_ch_inputs)
    
    Q = tf.concat(Q, axis=-1)
    K = tf.concat(K, axis=-1)
    V = tf.concat(V, axis=-1)
    #%%
    dims = Q.shape[-1]
    scale = dims ** -0.5
    attn = (Q @ tf.transpose(K,perm=[0,1,3,2])) * scale
    attn = tf.nn.softmax(attn,axis=-1)
    x = attn @ V
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(head_unit_num)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = vit_outputs + x
    
    return x
    
if __name__ == '__main__':
    one_ch_inputs = tf.keras.layers.Input(shape=(128, 6000,1))
    output = BoradAttentionViT(one_ch_inputs)
    model = tf.keras.Model(one_ch_inputs, output)
    model.summary()


        



    


# %%
