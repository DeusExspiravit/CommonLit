import tensorflow as tf
import keras
from keras import *

class CustomModel(layers.Layer):

    def __init__(self, num_heads, key_dim, dropout, k_initializer, out_units, in_units, hidden_units):
        super().__init__()
        self._params = {
            "num_heads": num_heads,
            "key_dim": key_dim,
            "dropout": dropout,
            "kernel_initializer": k_initializer
        }
        self._out_units = out_units
        self._in_units = in_units
    def build(self, input_shape):
        self._in_layer = layers.Dense(self._in_units, activation="relu")
        self._attn_1 = layers.MultiHeadAttention(**self._params)
        self._attn_2 = layers.MultiHeadAttention(**self._params)
        self._out_layer = layers.Dense(self._out_units, activation="relu")

    def call(self, key, query, value, *args, **kwargs):
