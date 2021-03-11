import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import Progbar
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from dalle_tensorflow.dalle_tensorflow import DiscreteVAE
from dalle_tensorflow.utils import normalize_tfds_img
import time


# Constants
IMG_SIZE = 128
BATCH_SIZE = 8

# Load and prepare CIFAR-10 data (image only)
ds_train = tfds.load('coco/2017', split='train', shuffle_files=True)
ds_train = ds_train.map(normalize_tfds_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.shuffle(BATCH_SIZE * 2).batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# DiscreteVAE Params
IMG_SIZE = 128
NUM_VISUAL_TOKENS = 8192
CODEBOOK_DIM = 512
NUM_LAYERS = 2
NUM_RESBLOCKS = 2
HIDDEN_DIM = 256
TEMPERATURE = 0.9
STRAIGHT_THROUGH = False

# Build the DiscreteVAE model
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

# Prepare for training
EPOCHS = 20
steps = len(ds_train)
# learning_rate = schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=steps, decay_rate=0.9)
learning_rate = 1e-3
optimizer = Adam(learning_rate=learning_rate)
loss_metric = Mean()
progress_bar = Progbar(target=steps, unit_name='step')

# Iterate over epochs.
for epoch in range(EPOCHS):
    # Extra line of printout because the ProgBar would overwrite the logs in the terminal
    print("Epoch: {}/{}".format(epoch + 1, EPOCHS))
    print("Epoch: {}/{}".format(epoch + 1, EPOCHS))
    start = time.time()

    # Iterate over the batches of the dataset.
    for step, x_batch in enumerate(ds_train):
        with tf.GradientTape() as tape:
            loss = vae(image=x_batch, return_recon_loss=True)

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients([(grad, var) for (grad, var) in
                                   zip(grads, vae.trainable_variables) if grad is not None])

        loss_metric(loss)
        progress_bar.update(step)

    end = time.time()
    time_per_step = (end - start) * 1000 / steps
    print(" - {:.3f}ms/step - loss: {:.6f}".format(time_per_step, loss_metric.result()))

    if (epoch + 1) % 5 == 0 and epoch != 0:
        vae.save_weights("./dalle_tensorflow/model_weights/vae/vae_weights" + "_" + str(epoch + 1))

# Save the model weights (subclassed model cannot use save_model)
vae.save_weights("./dalle_tensorflow/model_weights/vae/vae_weights")
