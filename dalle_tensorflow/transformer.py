import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dropout
from einops import rearrange

from dalle_tensorflow.reversible import ReversibleSequence, SequentialSequence
from dalle_tensorflow.layers import Linear, PreNorm, GEGLU
from dalle_tensorflow.utils import exists


class FeedForward(Layer):
    def __init__(self, input_dim, dropout=0.0, multiply=4.0):
        super(FeedForward, self).__init__()
        self.net = Sequential(layers=[
            Linear(input_dim * multiply * 2, input_dim),
            GEGLU(),
            Dropout(dropout),
            Linear(input_dim, input_dim * multiply)
        ])

    def call(self, x):
        return self.net(x)


class Attention(Layer):
    def __init__(self, input_dim, sequence_len, causal=True, heads=8, dim_head=64, dropout=0.0, noncausal_attn_len=0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.sequence_len = sequence_len
        self.scale = dim_head ** -0.5

        self.causal = causal
        self.noncausal_attn_len = noncausal_attn_len

        self.to_qkv = Linear(inner_dim * 3, input_dim, bias=False)
        self.to_out = Sequential(layers=[
            Linear(input_dim, inner_dim),
            Dropout(dropout)
        ])

    def call(self, x, mask=None):
        b, n, _ = x.shape
        h = self.heads
        qkv = tf.split(self.to_qkv(x), num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -tf.experimental.numpy.finfo(dots.dtype).max

        if exists(mask):
            shape = [mask.shape[0], 1, mask.shape[1], mask.shape[1]]
            m1 = tf.broadcast_to(input=rearrange(mask, 'b i -> b () i ()'), shape=shape)
            m2 = tf.broadcast_to(input=rearrange(mask, 'b j -> b () () j'), shape=shape)
            mask = tf.math.logical_and(m1, m2)
            dots = tf.where(~mask, mask_value, dots)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = tf.experimental.numpy.triu(tf.ones([i, j]), k=j - i + 1)
            mask = tf.cast(mask, dtype=tf.bool)

            if self.noncausal_attn_len > 1:
                index = slice(0, self.noncausal_attn_len)
                mask[index, index] = False

            dots = tf.where(mask, mask_value, dots)

        attn = tf.nn.softmax(dots, axis=-1)

        output = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.to_out(output)

        return output


class Transformer(Layer):
    def __init__(self, *, input_dim, depth, sequence_len, reversible=True, causal=True, heads=8, dim_head=64,
                 ff_multiply=4, attn_dropout=0.0, ff_dropout=0.0, noncausal_attn_len=0):
        super(Transformer, self).__init__()
        blocks = []

        for _ in range(depth):
            blocks.append([
                PreNorm(Attention(input_dim=input_dim, causal=causal, sequence_len=sequence_len, heads=heads,
                        dim_head=dim_head, dropout=attn_dropout, noncausal_attn_len=noncausal_attn_len)),
                PreNorm(FeedForward(input_dim=input_dim, multiply=ff_multiply, dropout=ff_dropout))
            ])

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}

        self.blocks = execute_type(blocks, args_route=attn_route_map)

    def call(self, x, **kwargs):
        return self.blocks(x, **kwargs)
