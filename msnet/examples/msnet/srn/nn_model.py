import tensorflow as tf

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Sequential, Model


def define_model():
    num = 1024
    inputs = Input(shape=(None, None, 256))
    x = Dense(num, activation='relu')(inputs)
    x = Dense(num, activation='relu')(x)
    x = Dense(num, activation='relu')(x)
    model = Model(inputs=inputs, outputs=x)
    return model

