import tensorflow as tf
from tensorflow.keras.layers import Layer

from operator import mul
from functools import reduce


# Referenced:
# https://github.com/lucidrains/axial-positional-embedding/blob/master/axial_positional_embedding/axial_positional_embedding.py
class AxialPositionalEmbedding(Layer):
    def __init__(self, dim, axial_shape, axial_dims=None):
        super(AxialPositionalEmbedding, self).__init__()

        self.dim = dim
        self.shape = axial_shape
        self.max_sequence_len = reduce(mul, axial_shape, 1)

        self.summed = axial_dims is None
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(
            axial_dims), 'Number of axial dimensions must equal the number of dimensions in the shape.'
        assert self.summed or not self.summed and sum(
            axial_dims) == dim, f'Axial dimensions must sum up to the target dimension {dim}.'

        self.embedding_weights = []

        for i, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[i] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            ax_embedding = tf.Variable(tf.random.normal(shape=ax_shape, mean=0.0, stddev=1.0))
            self.embedding_weights.append(ax_embedding)

    def call(self, x):
        b, t, e = x.shape
        assert t <= self.max_sequence_len, \
            f'Sequence length ({t}) must be less than the maximum sequence length allowed ({self.max_sequence_len}).'
        embeddings = []

        for ax_embedding in self.embedding_weights:
            axial_dim = ax_embedding.shape[-1]
            expand_shape = [b, *self.shape, axial_dim]
            embedding = tf.broadcast_to(input=ax_embedding, shape=expand_shape)
            embedding = tf.reshape(tensor=embedding, shape=[b, self.max_sequence_len, axial_dim])
            embeddings.append(embedding)

        pos_embedding = sum(embeddings) if self.summed else tf.concat(values=embeddings, axis=-1)
        return tf.cast(pos_embedding[:, :t], x.dtype)


# A mock parameter list object until below issue is resolved
class ParameterList(object):
    def __init__(self, kls, prefix, length):
        self.index = 0
        self.kls = kls
        self.prefix = prefix
        self.length = length

    def _keyname(self, prefix, index):
        return f'{prefix}_{index}'

    def append(self, x):
        setattr(self.kls, self._keyname(self.prefix, self.index), x)
        self.index += 1

    def to_list(self):
        return [getattr(self.kls, self._keyname(self.prefix, i)) for i in range(self.length)]


# Axial Positional Embedding for Images
class AxialPositionalEmbeddingImage(Layer):
    def __init__(self, dim, axial_shape, axial_dims=None):
        super(AxialPositionalEmbeddingImage, self).__init__()
        assert len(axial_shape) == 2, 'Axial shape must have 2 dimensions for images.'
        self.pos_embedding = AxialPositionalEmbedding(dim=dim, axial_shape=axial_shape, axial_dims=axial_dims)

    def call(self, image):
        b, h, w, c = image.shape
        image = tf.reshape(tensor=image, shape=[b, h * w, c])
        pos_embedding = self.pos_embedding(image)
        return tf.reshape(tensor=pos_embedding, shape=[b, h, w, c])
