import tensorflow as tf
from inspect import isfunction


def exists(value):
    return value is not None


def is_unique(array):
    return {element: True for element in array}.keys()


def default(value, d):
    if exists(value):
        return value
    return d() if isfunction(d) else d


def cast_tuple(value, depth):
    return value if isinstance(value, tuple) else (value,) * depth


def always(value):
    def inner(*args, **kwargs):
        return value
    return inner


def is_empty(tensor):
    return tensor.shape == 0


def masked_mean(input_tensor, mask, axis=1):
    tensor = tf.where(~mask[:, :, None], 0., input_tensor)
    return tf.math.reduce_sum(tensor, axis=axis) / tf.math.reduce_sum(mask, axis=axis)[..., None]


def top_k(logits, threshold=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - threshold) * num_logits), 1)
    top_k_values, top_k_indices = tf.math.top_k(logits, k=k)

    # Indices are only columns, we will stack it so the row indices is also there and make tensor of row numbers
    num_rows = logits.shape[0]
    row_range = tf.range(num_rows)
    row_tensor = tf.tile(row_range[:, None], (1, k))

    top_k_full_indices = tf.stack([row_tensor, top_k_indices], axis=2)

    mask = tf.scatter_nd(top_k_full_indices, tf.ones([num_rows, k]), logits.shape.as_list())
    mask = tf.dtypes.cast(mask, tf.bool)
    probs = tf.where(~mask, float('-inf'), logits)
    return probs


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, axis, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature, axis=axis)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def normalize_tfds_img(data):
    data['image'] = tf.image.resize_with_pad(image=data['image'], target_height=128, target_width=128)
    data['image'] = tf.cast(data['image'], tf.float32) / 255.
    return data['image']


def normalize_img(data):
    return tf.cast(data, tf.float32) / 255.


def byte_to_string(byte_string):
    return byte_string.decode("utf-8")
