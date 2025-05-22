# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.metrics import Precision, Recall
import time
import tensorflow as tf
from tensorflow.keras.layers import Layer


#----------------
# use following SparseMaskedLSTMCell class and call to replace Vanilla LSTM and corresponding call
#----------------

class SparseMaskedLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, kernel_mask, recurrent_kernel_mask, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]
        self.output_size = units
        self.activation = tf.keras.activations.get("tanh")
        self.recurrent_activation = tf.keras.activations.get("sigmoid")

        # Tile to 4 gates
        self.kernel_mask_np = np.tile(kernel_mask, (1, 4)).astype(np.float32)  # (input_dim, 4*units)
        self.recurrent_mask_np = np.tile(recurrent_kernel_mask, (1, 4)).astype(np.float32)  # (units, 4*units)

    def build(self, input_shape):
        input_dim = input_shape[-1]
    
        def get_sparse_components(mask_np, name_prefix):
            indices = np.array(np.nonzero(mask_np)).astype(np.int64).T  # (N, 2)
            values = self.add_weight(
                name=f"{name_prefix}_values",
                shape=(indices.shape[0],),
                initializer="random_normal",
                trainable=True
            )
            dense_shape = [mask_np.shape[0], mask_np.shape[1]]
            return tf.constant(indices, dtype=tf.int64), values, dense_shape
    
        self.k_indices, self.k_values, self.k_shape = get_sparse_components(self.kernel_mask_np, "k")
        self.rk_indices, self.rk_values, self.rk_shape = get_sparse_components(self.recurrent_mask_np, "rk")
    
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            initializer="zeros",
            name="bias"
        )


    def call(self, inputs, states):
        h_tm1, c_tm1 = states

        # Sparse kernel (input → gates)
        sparse_kernel = tf.sparse.SparseTensor(
            indices=self.k_indices,
            values=self.k_values,
            dense_shape=self.k_shape
        )
        sparse_kernel = tf.sparse.reorder(sparse_kernel)
        z = tf.sparse.sparse_dense_matmul(inputs, sparse_kernel)

        # Sparse recurrent kernel (hidden → gates)
        sparse_recurrent = tf.sparse.SparseTensor(
            indices=self.rk_indices,
            values=self.rk_values,
            dense_shape=self.rk_shape
        )
        sparse_recurrent = tf.sparse.reorder(sparse_recurrent)
        z += tf.sparse.sparse_dense_matmul(h_tm1, sparse_recurrent)

        z += self.bias
        i, f, c_hat, o = tf.split(z, num_or_size_splits=4, axis=1)

        i = self.recurrent_activation(i)
        f = self.recurrent_activation(f)
        c_hat = self.activation(c_hat)
        o = self.recurrent_activation(o)

        c = f * c_tm1 + i * c_hat
        h = o * self.activation(c)
        return h, [h, c]
    
    
def create_custom_kernel_mask():
    kernel_mask = np.zeros((322, 96), dtype=np.float32)
    blocks = [
        (0, 0, 28, 32),
        (28, 32, 98, 32),
        (28+98, 64, 196, 32)
    ]
    for (start_row, start_col, height, width) in blocks:
        kernel_mask[start_row:start_row+height, start_col:start_col+width] = 1
    return kernel_mask

def create_recurrent_kernel_mask(units, block_size):
    recurrent_kernel_mask = np.zeros((units, units), dtype=np.float32)
    num_blocks = units // block_size
    for i in range(num_blocks):
        start_index = i * block_size
        end_index = start_index + block_size
        recurrent_kernel_mask[start_index:end_index, start_index:end_index] = 1
    return recurrent_kernel_mask

# 创建掩码示例
kernel_mask = create_custom_kernel_mask()
recurrent_kernel_mask = create_recurrent_kernel_mask(units=96, block_size=32)


# Generate masks
input_dim = 322  # Dimension of input features
units = 96     # Number of units in the RNN cell


input_layer = tf.keras.Input(shape=(360, input_dim))
cell = SparseMaskedLSTMCell(units=96, kernel_mask=kernel_mask, recurrent_kernel_mask=recurrent_kernel_mask)
rnn_layer = tf.keras.layers.RNN(cell, return_sequences=False)
rnn_output = rnn_layer(input_layer)
output_layer = tf.keras.layers.Dense(4, activation='softmax')(rnn_output)
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)


# ----------------------
# Get the trained weights from the model, use this to replace RNN_kernel=lstm_weights[0], RNN_rekernel=lstm_weights[1]
# ----------------------
cellll = model.layers[1].cell  # Access your custom SparseMaskedLSTMCell

# Convert tensors to numpy
k_indices_np = cellll.k_indices.numpy()
k_values_np = cellll.k_values.numpy()
k_shape_np = cellll.k_shape

rk_indices_np = cellll.rk_indices.numpy()
rk_values_np = cellll.rk_values.numpy()
rk_shape_np = cellll.rk_shape


np.savez("sparse_lstm_weights.npz",
         k_indices=k_indices_np,
         k_values=k_values_np,
         k_shape=k_shape_np,
         rk_indices=rk_indices_np,
         rk_values=rk_values_np,
         rk_shape=rk_shape_np)
#########################




