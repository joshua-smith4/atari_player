import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_outputs):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (8, 8), padding="same", input_shape=input_shape, strides=4, activation='relu'))
    model.add(layers.Conv2D(32, (4, 4), padding="same", strides=2, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_outputs, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    return model

