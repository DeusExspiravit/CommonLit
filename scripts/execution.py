import sys

sys.path.extend(["/Users/arvinprince/Tensorflow/CommonLit/scripts"])

import tensorflow as tf
import keras
from keras import *
from keras.preprocessing.text import Tokenizer
import pandas as pd
from processing import *

df_PT = pd.read_csv("/Users/arvinprince/Tensorflow/CommonLit/data/prompts_train.csv")
df_ST = pd.read_csv("/Users/arvinprince/Tensorflow/CommonLit/data/summaries_train.csv")

parsing = Distiller(num_words=10000, oov_token="<OOV>", lower=False)
ds = parsing((df_PT, df_ST))

pos_enc = PositionalEmbedding(10000, 985, 256,
                              7165, 32, tf.float32)

dummy = pos_enc(ds[1])