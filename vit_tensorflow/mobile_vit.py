import tensorflow as tf

from keras.src.applications import imagenet_utils
# For versions <TF2.13 change the above import to:
# from keras.applications import imagenet_utils

from keras import layers

import numpy as np

# Values are from table 4..

expansion_factor = 2  # expansion factor for the MobileNetV2 blocks.

def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(
        filters, kernel_size = kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )
    return conv_layer(x)


# Reference: https://git.io/JKgtC


def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    if strides == 2:
        m = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(m, 3))(m)
    m = layers.DepthwiseConv2D(
        3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m


# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]], dropout_rate=0.1,)
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, patch_shape, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )
    patch_h, patch_w = patch_shape
    patch_size = patch_h * patch_w
    # Unfold into patches and then pass through Transformers.
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)
    non_overlapping_patches = tf.image.extract_patches(
        images = local_features,
        sizes=[1, patch_h, patch_w, 1],
        strides=[1,patch_h, patch_w, 1],
        rates=[1,1,1,1],
        padding='VALID'
        )
    non_overlapping_patches = layers.Reshape((num_patches, patch_size, projection_dim))(
        non_overlapping_patches
    )
    #non_overlapping_patches = tf.transpose(non_overlapping_patches, [0,2,1,3])
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features


def one_channel_mobile_vit(image_size, patch_shape = (2,2)):
    image_h, image_w, image_c = image_size
    inputs = tf.keras.Input((image_h, image_w, image_c))
    #x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = inputs
    # Initial conv-stem -> MV2 block.
    # (128, 2000)
    x = conv_block(x, filters=16)
    # (64, 1000)
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=16
    )

    # Downsampling with MV2 block.
    # (64, 1000)
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=24, strides=2
    )
    # (32, 500)
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )
    
    # First MV2 -> MobileViT block.
    # (32, 500)
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=32, strides=2
    )
    x = mobilevit_block(x, patch_shape, num_blocks=2, projection_dim=64)
    
    # Second MV2 -> MobileViT block.

    x = inverted_residual_block(
        x, expanded_channels=64 * expansion_factor, output_channels=48, strides=2
    )
    x = mobilevit_block(x, patch_shape, num_blocks=4, projection_dim=80)
    # (16, 250)
    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=80 * expansion_factor, output_channels=64, strides=1
    )
    x = mobilevit_block(x, patch_shape, num_blocks=3, projection_dim=96)
    x = conv_block(x, filters=320, kernel_size=1, strides=1)

    # Classification head.
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)