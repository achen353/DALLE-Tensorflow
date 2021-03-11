import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, smart_resize, save_img
from dalle_tensorflow.dalle_tensorflow import DiscreteVAE
from dalle_tensorflow.utils import normalize_img
import numpy as np
from os import listdir
from os.path import isfile, join

# Using the first image in the folder to make inference
IMAGE_ROOT = "./dalle_tensorflow/data/images/"
image_path = np.array([join(IMAGE_ROOT, file) for file in listdir(IMAGE_ROOT) if isfile(join(IMAGE_ROOT, file))])[0]

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

vae.load_weights("./dalle_tensorflow/model_weights/vae/vae_weights")

image = load_img(image_path)
image = np.array(image)
image = smart_resize(image, size=[IMG_SIZE, IMG_SIZE])
image = normalize_img(image)
image = tf.expand_dims(image, axis=0)

output = vae(image)
output = tf.reshape(tensor=output, shape=[IMG_SIZE, IMG_SIZE, 3])
output = save_img(path="vae_out.jpg", x=output)
