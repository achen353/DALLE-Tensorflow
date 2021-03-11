import tensorflow as tf
from tensorflow.keras.layers import Layer


# For routing arguments into the functions of the reversible layer
def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args


# Referenced https://github.com/cerebroai/revflow/blob/master/revflow/blocks.py
class ReversibleBlock(Layer):
    def __init__(self, f_block, g_block, split_along_axis=1):
        super(ReversibleBlock, self).__init__()
        self.axis = split_along_axis
        self.f = f_block
        self.g = g_block

    def call(self, x, f_args={}, g_args={}):
        """Apply residual block to inputs."""
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=self.axis)
        f_x2 = self.f(x2, **f_args)
        y1 = x1 + f_x2
        g_y1 = self.g(y1, **g_args)
        y2 = x2 + g_y1
        return tf.concat([y1, y2], axis=self.axis)

    # def backward_grads_and_vars(self, y, dy):
    #     """Manually compute backward gradients given input and output grads."""
    #     dy1, dy2 = tf.split(dy, num_or_size_splits=2, axis=self.axis)
    #
    #     with tf.GradientTape(persistent=True) as tape:
    #         y = tf.identity(y)
    #         tape.watch(y)
    #         y1, y2 = tf.split(y, num_or_size_splits=2, axis=self.axis)
    #         z1 = y1
    #         gz1 = self.g(z1)
    #         x2 = y2 - gz1
    #         fx2 = self.f(x2)
    #         x1 = z1 - fx2
    #
    #         grads_combined = tape.gradient(
    #             gz1, [z1] + self.g.trainable_variables, output_gradients=dy2)
    #         dz1 = dy1 + grads_combined[0]
    #         dg = grads_combined[1:]
    #         dx1 = dz1
    #
    #         grads_combined = tape.gradient(
    #             fx2, [x2] + self.f.trainable_variables, output_gradients=dz1)
    #         dx2 = dy2 + grads_combined[0]
    #         df = grads_combined[1:]
    #
    #         del tape
    #
    #     grads = df + dg
    #     vars_ = self.f.trainable_variables + self.g.trainable_variables
    #
    #     x = tf.concat([x1, x2], axis=self.axis)
    #     dx = tf.concat([dx1, dx2], axis=self.axis)
    #
    #     return x, dx, grads, vars_


class SequentialSequence(Layer):
    def __init__(self, blocks, args_route={}):
        super(SequentialSequence, self).__init__()
        assert all(len(route) == len(blocks) for route in args_route.values()), \
            'Each argument route map must have the same depth as the number of sequential layers.'
        self.blocks = blocks
        self.args_route = args_route

    def call(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.blocks))
        blocks_and_args = list(zip(self.blocks, args))

        for (f, g), (f_args, g_args) in blocks_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


class ReversibleSequence(tf.keras.layers.Layer):
    """Single reversible block containing several `_Residual` blocks.
    Each `_Residual` block in turn contains two _ResidualInner blocks,
    corresponding to the `F`/`G` functions in the paper.
    This is based on PyTorch's RevTorch - ReversibleSequence
    """

    def __init__(self, blocks, args_route={}):
        super(ReversibleSequence, self).__init__()
        self.args_route = args_route
        self.blocks = [ReversibleBlock(f_block=f_block, g_block=g_block) for f_block, g_block in blocks]

    def call(self, x, **kwargs):
        """Apply reversible block to inputs."""
        x = tf.concat([x, x], axis=1)
        args = route_args(self.args_route, kwargs, len(self.blocks))
        args = list(map(lambda h: {'f_args': h[0], 'g_args': h[1]}, args))
        for block, kwarg in zip(self.blocks, args):
            x = block(x, **kwarg)
        x = tf.stack(tf.split(x, num_or_size_splits=2, axis=1))
        x = tf.reduce_mean(input_tensor=x, axis=0)
        return x

    # def backward_grads_and_vars(self, x, y, dy):
    #     """Apply reversible block backward to outputs."""
    #     grads_all = []
    #     vars_all = []
    #
    #     for i in reversed(range(len(self.blocks))):
    #         block = self.blocks[i]
    #         if i == 0:
    #             # First block usually contains downsampling that can't be reversed
    #             with tf.GradientTape() as tape:
    #                 x = tf.identity(x)
    #                 tape.watch(x)
    #                 y = block(x)
    #
    #                 grads_combined = tape.gradient(
    #                     y, [x] + block.trainable_variables, output_gradients=dy)
    #                 dy = grads_combined[0]
    #                 grads_all += grads_combined[1:]
    #                 vars_all += block.trainable_variables
    #         else:
    #             y, dy, grads, vars_ = block.backward_grads_and_vars(y, dy)
    #             grads_all += grads
    #             vars_all += vars_
    #
    #     return dy, grads_all, vars_all
