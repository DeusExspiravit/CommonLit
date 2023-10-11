import tensorflow as tf
import keras
from keras import *
from keras.preprocessing.text import Tokenizer
import numpy as np

class Distiller(layers.Layer):
    def __init__(self,
                 num_words=1000,
                 oov_token="OOV",
                 lower=False,
                 ):
        super().__init__()
        self.tokenizer_params = {"num_words": num_words,
                                 "oov_token": oov_token,
                                 "lower": lower
                                 }
    def build(self, input_shape):
        self.tokenizer = Tokenizer(**self.tokenizer_params)
        self.long_list = ["prompt_text", "text"]
        self.short_list = ["prompt_question", "prompt_title"]

    def call(self, inputs, *args, **kwargs):

        if not isinstance(inputs, tuple):
            raise TypeError("The input should be a collection of two DataFrames")

        df = self.merger(inputs[0], inputs[1])
        columns = self.filtered_col(df)
        ds = []
        ds_q = []

        for x in columns:
            self.tokenizer.fit_on_texts(df[x])
            y = self.tokenizer.texts_to_sequences(df[x])
            if x in self.short_list:
                ds_q.append(supplementary_func(27)(y))
            else:
                ds.append(supplementary_func(985)(y))

        ds, ds_q = tf.convert_to_tensor(ds, dtype=tf.float32), tf.convert_to_tensor(ds_q, dtype=tf.float32)
        target = tf.convert_to_tensor(df.loc[:, ["content", "wording"]], dtype=tf.float32)

        return ds_q, ds, target

    def merger(self, df1, df2):
        return df1.merge(df2)

    def filtered_col(self, df):
        col = df.select_dtypes(object).columns
        col = [c for c in col if c[-2:] != "id"]
        return col

class supplementary_func:

    def __init__(self, max_len,):
        self.max_len = max_len

    def __call__(self,inputs, *args, **kwargs):
        text = self.to_rectangular(inputs)
        return self.pad(text, self.max_len)

    def pad(self, seq, max_len):
        return tf.pad(seq, [[0,0], [0, max_len - tf.shape(seq)[-1]]])

    def to_rectangular(self, ragged_list):
        return tf.ragged.constant(ragged_list, dtype=tf.float32).to_tensor()


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self,
                 vocab_size: int,
                 max_len: int,
                 embed_dim: int,
                 num_phrases:int,
                 batch_size:int,
                 dtype = tf.int32):
        super().__init__()
        self.d_model = embed_dim
        self.embedding = keras.layers.Embedding(vocab_size, embed_dim)
        self.type = dtype
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_phrases = num_phrases
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x, *args, **kwargs):
        # pos_enc = self.positional_encoding(num_phrases=self.num_phrases,
        # length=self.max_len, depth=self.vocab_size, dtype=self.type)
        dim = tf.shape(x)[-1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, dtype=self.type))
        pos_emb = tf.cast(tf.range(start=1, limit=self.embed_dim+1), dtype=tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :]
        # x = pos_enc[tf.newaxis, :, :, :self.emb_dim] + x
        return x + pos_emb
    def positional_encoding(self, num_phrases, length, depth, dtype):
        depth = int(depth / 2)

        positions = np.arange(num_phrases*length).reshape((num_phrases, length))[:, :, np.newaxis]
        depths = np.arange(int(length*depth)).reshape((length, depth))[np.newaxis, :, :] / depth

        angle_rates = 1 / (1e4 ** depths)
        angle_rads = positions * angle_rates

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1,
        )
        return tf.cast(pos_encoding, dtype=dtype)

    def dataset_generator(inputs, batch_size):
        return tf.data.Dataset.from_tensor_slices(inputs, name="llm_sc").batch(batch_size)
