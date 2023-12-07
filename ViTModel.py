#%%
import keras
from vit_tensorflow.vit import ViT
import tensorflow as tf
import numpy as np

def BuildMViTModel(full_ch_inputs):
    vit_model_outputs = []
    
    channel_num = full_ch_inputs.shape[1]
    
    for i in range(channel_num):
        one_channel_input = full_ch_inputs[:,i]
        
        v = ViT(
            image_shape = (full_ch_inputs.shape[-3], full_ch_inputs.shape[-2], full_ch_inputs.shape[-1]),
            patch_shape = (50, 50),
            projection_dim = 64,
            num_heads = 4,
            transformer_units = [64],
            mlp_head_units = [32],
            num_layers = 4,
        )
        
        single_vit_output = v.create_vit_layer(inputs = one_channel_input, name = f'vit{i+1}')
        vit_model_outputs.append(single_vit_output)

    features_concated = tf.keras.layers.concatenate(vit_model_outputs)
    x = tf.keras.layers.Dense(256, activation=keras.activations.gelu)(features_concated)
    outputs = tf.keras.layers.Dense(2,activation=keras.activations.softmax)(x)
    
    return keras.models.Model(inputs = full_ch_inputs, outputs = outputs, name = 'mvit')


# %%
