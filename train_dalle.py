import tensorflow as tf
from tensorflow.keras.utils import Progbar
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.preprocessing.image import load_img, smart_resize
from dalle_tensorflow.dalle_tensorflow import DiscreteVAE, DALLE
from dalle_tensorflow.utils import normalize_img

import time
import numpy as np
from os import listdir
from os.path import isfile, join

from bpemb import BPEmb


def shuffle_data(images, captions):
    rand_idx = np.arange(images.shape[0])
    np.random.shuffle(rand_idx)

    return images[rand_idx], captions[rand_idx]


# Constants
IMAGE_ROOT = "./dalle_tensorflow/data/images/"
CAPTION_ROOT = "./dalle_tensorflow/data/captions/"

# Set up paths to images and captions
image_paths = np.array([join(IMAGE_ROOT, file) for file in listdir(IMAGE_ROOT) if isfile(join(IMAGE_ROOT, file))])
caption_paths = np.array([join(CAPTION_ROOT, file) for file in listdir(CAPTION_ROOT)
                          if isfile(join(CAPTION_ROOT, file))])
image_paths.sort()
caption_paths.sort()

image_paths, caption_paths = shuffle_data(image_paths, caption_paths)

# DiscreteVAE Params
IMG_SIZE = 128
NUM_VISUAL_TOKENS = 8192
CODEBOOK_DIM = 512
NUM_LAYERS = 2
NUM_RESBLOCKS = 2
HIDDEN_DIM = 256
TEMPERATURE = 0.9
STRAIGHT_THROUGH = False

# Build our model and load DiscreteVAE with pre-trained weights
vae = DiscreteVAE(
    image_size=IMG_SIZE,                # Size of image
    num_tokens=NUM_VISUAL_TOKENS,       # Number of visual tokens: The paper used 8192, but could be smaller for downsized projects
    codebook_dim=CODEBOOK_DIM,          # Codebook dimension
    num_layers=NUM_LAYERS,              # Number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
    num_resblocks=NUM_RESBLOCKS,        # Number of resnet blocks
    hidden_dim=HIDDEN_DIM,              # Hidden dimension
    temperature=TEMPERATURE,            # Gumbel softmax temperature. The lower this is, the harder the discretization
    straight_through=STRAIGHT_THROUGH   # Straight-through for gumbel softmax. unclear if it is better one way or the other
)

vae.load_weights("./dalle_tensorflow/model_weights/vae/vae_weights")

# DALLE Params
VOCAB_SIZE = 10000
assert VOCAB_SIZE in BPEmb.available_vocab_sizes('en'), "Vocab size not available. " \
                                                        "Call `BPEmb.available_vocab_sizes('en')` to check sizes"
TEXT_SEQ_LEN = 128
DEPTH = 24
HEADS = 8
DIM_HEAD = 64
REVERSIBLE = True
ATTN_DROPOUT = 0.1
FF_DROPOUT = 0.1

# Build our DALL-E model
dalle = DALLE(
    dim=CODEBOOK_DIM,                           # Codebook Dimension
    vae=vae,                                    # DiscreteVAE instance: image sequence length and number of image tokens inferred
    num_text_tokens=VOCAB_SIZE + 1,             # Vocab size for text. Add 1 for <PAD>
    text_sequence_len=TEXT_SEQ_LEN,             # Text sequence length
    depth=DEPTH,                                   # Transformer depth: should aim to be 64
    heads=HEADS,                                   # Attention heads
    dim_head=DIM_HEAD,                                # Attention head dimension
    reversible=REVERSIBLE,                            # Whether to use ReversibleSequence or SequentialSequence
    attn_dropout=ATTN_DROPOUT,                           # Attention dropout
    ff_dropout=FF_DROPOUT                              # Feedforward dropout
)

# Prepare for training
bpe_encoder = BPEmb(lang="en", vs=VOCAB_SIZE, add_pad_emb=True)
EPOCHS = 20
BATCH_SIZE = 2
steps = (np.ceil(len(image_paths) / float(BATCH_SIZE))).astype(np.int)
learning_rate = schedules.ExponentialDecay(initial_learning_rate=4e-3, decay_steps=steps, decay_rate=0.95)
optimizer = Adam(learning_rate=learning_rate)
loss_metric = Mean()
progress_bar = Progbar(target=steps, unit_name='step')

# Iterate over epochs.
for epoch in range(EPOCHS):
    image_paths, caption_paths = shuffle_data(image_paths, caption_paths)

    # Extra line of printout because the ProgBar would overwrite the logs in the terminal
    print("Epoch: {}/{}".format(epoch + 1, EPOCHS))
    print("Epoch: {}/{}".format(epoch + 1, EPOCHS))
    start = time.time()

    # Iterate over the batches of the dataset.
    for step in range(steps):
        batch_image_paths = image_paths[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
        batch_cap_paths = caption_paths[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]

        batch_image = []
        batch_cap = []
        for i in range(len(batch_cap_paths)):
            image_path = batch_image_paths[i]
            cap_path = batch_cap_paths[i]

            image = load_img(image_path)
            image = np.array(image)
            image = smart_resize(image, size=[IMG_SIZE, IMG_SIZE])
            image = normalize_img(image)

            with open(cap_path) as f:
                text = f.read()
                f.close()
            text = bpe_encoder.encode_ids(text)
            text = np.array(text)
            text = np.pad(array=text, pad_width=[0, TEXT_SEQ_LEN - len(text)])

            # print(image.shape)
            batch_image.append(image)
            batch_cap.append(text)

        batch_image = np.stack(batch_image)
        batch_cap = np.stack(batch_cap)
        batch_image = tf.convert_to_tensor(batch_image, dtype=tf.float64)
        batch_cap = tf.convert_to_tensor(batch_cap, dtype=tf.int64)
        batch_mask = tf.cast(tf.where(batch_cap != 0, 1, 0, batch_cap), dtype=tf.bool)

        with tf.GradientTape() as tape:
            loss = dalle(text=batch_cap, image=batch_image, mask=batch_mask, return_loss=True)

        grads = tape.gradient(loss, dalle.trainable_weights)
        optimizer.apply_gradients([(grad, var) for (grad, var) in
                                   zip(grads, dalle.trainable_variables) if grad is not None])
        loss_metric(tf.reduce_mean(tf.reduce_sum(loss, axis=-1)))
        progress_bar.update(step)

    end = time.time()
    time_per_step = (end - start) * 1000 / steps
    print(" - {:.3f}ms/step - loss: {:.6f}".format(time_per_step, loss_metric.result()))

    if (epoch + 1) % 5 == 0 and epoch != 0:
        dalle.save_weights("./dalle_tensorflow/model_weights/dalle/dalle_weights" + "_" + str(epoch + 1))

# Save the model weights (subclassed model cannot use save_model)
dalle.save_weights("./dalle_tensorflow/model_weights/dalle/dalle_weights")
