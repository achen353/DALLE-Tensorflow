import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization


class Linear(Layer):
    def __init__(self, units, input_dim, bias=True):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            name="w", shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(name="b", shape=(units,), initializer="zeros", trainable=bias)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class PreNorm(Layer):
    def __init__(self, fn):
        super(PreNorm, self).__init__()
        self.norm = LayerNormalization()
        self.fn = fn

    def call(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(Layer):
    def call(self, x):
        x, gates = tf.split(x, num_or_size_splits=2, axis=-1)
        return x * tf.nn.gelu(gates)
