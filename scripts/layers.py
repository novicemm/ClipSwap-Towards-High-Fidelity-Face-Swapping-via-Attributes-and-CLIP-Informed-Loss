import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Layer


tf.keras.utils.disable_interactive_logging()

class AdaIN(Layer):
    def __init__(self, **kwargs):
        super(AdaIN, self).__init__(**kwargs)

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]

        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]

        self.dense_1 = Dense(self.x_channels)
        self.dense_2 = Dense(self.x_channels)

    def call(self, inputs):
        x, w = inputs
        ys = tf.reshape(self.dense_1(w), (-1, 1, 1, self.x_channels))
        yb = tf.reshape(self.dense_2(w), (-1, 1, 1, self.x_channels))
        return ys * x + yb

    def get_config(self):
        config = {
            #'w_channels': self.w_channels,
            #'x_channels': self.x_channels
        }
        base_config = super(AdaIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class AdaptiveAttention(Layer):
    def __init__(self, **kwargs):
        super(AdaptiveAttention, self).__init__(**kwargs)

    def call(self, inputs):
        m, a, i = inputs
        return (1 - m) * a + m * i

    def get_config(self):
        base_config = super(AdaptiveAttention, self).get_config()
        return base_config

