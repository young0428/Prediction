"""
Title: Image classification with Vision Transformer
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/01/18
Last modified: 2021/01/18
Description: Implementing the Vision Transformer (ViT) model for image classification.
Accelerator: GPU
"""

"""
## Introduction

This example implements the [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
model by Alexey Dosovitskiy et al. for image classification,
and demonstrates it on the CIFAR-100 dataset.
The ViT model applies the Transformer architecture with self-attention to sequences of
image patches, without using convolution layers.

"""

"""
## Setup
"""

import os

#os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]

import tensorflow as tf
import tensorflow.keras.layers as layers
import time


import numpy as np
import matplotlib.pyplot as plt

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = projection_dim
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        
        positions = tf.keras.backend.expand_dims(
            tf.keras.backend.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = layers.Dense(self.projection)(patch)
        encoded = projected_patches + self.position_embedding(positions)

        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
    
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = tf.keras.backend.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size[0]
        num_patches_w = width // self.patch_size[1]
        patch_h = self.patch_size[0]
        patch_w = self.patch_size[1]
        
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_h, patch_w, 1],
            strides=[1,patch_h, patch_w, 1],
            rates=[1,1,1,1],
            padding='VALID'
            )
        
        patches = tf.keras.backend.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size[0] * self.patch_size[1] * channels,
            ),
        )
        
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class ViT :
    def __init__(self, image_shape, patch_shape,  num_heads, transformer_units, mlp_head_units, projection_dim = 64, num_layers = 8, learning_rate = 0.001, sweight_decay = 0.0001, num_classes = 1):
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.num_classes = num_classes
        self.input_shape = image_shape
        self.patch_shape = patch_shape
        self.num_patches = int((image_shape[0] * image_shape[1])  / (patch_shape[0] * patch_shape[1]))
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.transformer_layers = num_layers
        self.mlp_head_units = mlp_head_units

    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=keras.activations.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def create_vit_layer(self, inputs, name = 'ViT'):
        # Create patches.
        
        patches = Patches(self.patch_shape)(inputs)

        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)
        time.sleep(3)
        Q = []
        K = []
        V = []
        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_layer = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )
            
            attention_output, query, key, value = attention_layer(x1,x1)
            
            Q.append(query)
            K.append(key)
            V.append(value)
            
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        # logits = layers.Dense(self.num_classes)(features)
        # Create the Keras model.
        #model = keras.Model(inputs=inputs, outputs=features, name=name)

        return features, Q, K, V


"""
## Compile, train, and evaluate the mode
"""







"""
After 100 epochs, the ViT model achieves around 55% accuracy and
82% top-5 accuracy on the test data. These are not competitive results on the CIFAR-100 dataset,
as a ResNet50V2 trained from scratch on the same data can achieve 67% accuracy.

Note that the state of the art results reported in the
[paper](https://arxiv.org/abs/2010.11929) are achieved by pre-training the ViT model using
the JFT-300M dataset, then fine-tuning it on the target dataset. To improve the model quality
without pre-training, you can try to train the model for more epochs, use a larger number of
Transformer layers, resize the input images, change the patch size, or increase the projection dimensions.
Besides, as mentioned in the paper, the quality of the model is affected not only by architecture choices,
but also by parameters such as the learning rate schedule, optimizer, weight decay, etc.
In practice, it's recommended to fine-tune a ViT model
that was pre-trained using a large, high-resolution dataset.
"""