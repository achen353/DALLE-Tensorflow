from math import log2, sqrt

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv2D, ZeroPadding2D, Embedding, Conv2DTranspose, LayerNormalization
from tensorflow.keras.utils import normalize

from einops import rearrange
from dalle_tensorflow.utils import exists, is_empty, masked_mean, top_k, gumbel_softmax
from dalle_tensorflow.layers import Linear
from dalle_tensorflow.transformer import Transformer
from dalle_tensorflow.axial_positional_embedding import AxialPositionalEmbedding


class ResBlock(Layer):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.net = Sequential([
            ZeroPadding2D(padding=(1, 1)),
            Conv2D(filters, 3, padding='valid', activation='relu'),
            ZeroPadding2D(padding=(1, 1)),
            Conv2D(filters, 3, padding='valid', activation='relu'),
            Conv2D(filters, 1)
        ])

    def call(self, x):
        return self.net(x) + x


class DiscreteVAE(Model):
    def __init__(self, image_size=256, num_tokens=512, codebook_dim=512, num_layers=3, num_resblocks=0, hidden_dim=64,
                 channels=3, temperature=0.9, straight_through=False):
        super(DiscreteVAE, self).__init__()
        assert log2(image_size).is_integer(), 'Image size must be a power of 2.'
        assert num_layers >= 1, 'Number of layers must at least 1.'

        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = Embedding(num_tokens, codebook_dim)

        self.codebook.build([None, self.num_tokens])

        encode_channels = [hidden_dim] * num_layers
        decode_channels = list(reversed(encode_channels))

        encode_channels = [channels, *encode_channels]

        decode_initial_channels = codebook_dim if not num_resblocks > 0 else decode_channels[0]
        decode_channels = [decode_initial_channels, *decode_channels]

        encode_channels_io, decode_channels_io = map(
            lambda tensor: list(zip(tensor[:-1], tensor[1:])), (encode_channels, decode_channels))

        encode_layers = []
        decode_layers = []

        for (encode_in, encode_out), (decode_in, decode_out) in zip(encode_channels_io, decode_channels_io):
            encode_layers.append(Sequential([
                ZeroPadding2D(padding=(1, 1)),
                Conv2D(encode_out, 4, strides=(2, 2), padding='valid', activation='relu')
            ]))
            decode_layers.append(Sequential([
                Conv2DTranspose(decode_out, 4, strides=(2, 2), padding='same', activation='relu'),
            ]))

        for _ in range(num_resblocks):
            decode_layers.insert(0, ResBlock(decode_channels[1]))
            encode_layers.append(ResBlock(encode_channels[-1]))

        if num_resblocks:
            decode_layers.insert(0, Conv2D(decode_channels[1], 1))

        encode_layers.append(Conv2D(num_tokens, 1))
        decode_layers.append(Conv2D(channels, 1))

        self.encoder = Sequential(encode_layers)
        self.decoder = Sequential(decode_layers)

    def get_codebook_indices(self, images):
        logits = tf.stop_gradient(self.call(images, return_logits=True))
        logits = tf.math.argmax(logits, axis=-1)
        codebook_indices = tf.reshape(logits, [-1, tf.reduce_prod(logits.shape[1:])])
        return codebook_indices

    def decode(self, image_sequence):
        image_embeddings = self.codebook(image_sequence)
        b, n, d = image_embeddings.shape
        h = w = int(sqrt(n))

        image_embeddings = rearrange(image_embeddings, 'b (h w) d -> b h w d', h=h, w=w)
        images = self.decoder(image_embeddings)
        return images

    def call(self, image, return_recon_loss=False, return_logits=False):
        logits = self.encoder(image)

        if return_logits:
            # Return logits for getting hard image indices for DALL-E training
            return logits

        soft_one_hot = gumbel_softmax(logits=logits, temperature=self.temperature, axis=-1, hard=self.straight_through)

        codebook_weights = tf.convert_to_tensor(self.codebook.get_weights())
        codebook_weights = tf.squeeze(input=codebook_weights, axis=0)
        sampled = tf.einsum('b h w n, n d -> b h w d', soft_one_hot, codebook_weights)
        out = self.decoder(sampled)

        if not return_recon_loss:
            return out
        loss = tf.compat.v1.losses.mean_squared_error(image, out)
        return loss


class CLIP(Layer):
    def __init__(self, *, text_dim=512, image_dim=512, latent_dim=512, num_text_tokens=10000, text_encode_depth=6,
                 text_sequence_len=256, text_heads=8, num_visual_tokens=512, visual_encode_depth=6, visual_heads=8,
                 visual_image_size=256, visual_patch_size=32, channels=3):
        super(CLIP, self).__init__()
        self.text_embedding = Embedding(num_text_tokens, text_dim)
        self.text_pos_embedding = Embedding(text_sequence_len, text_dim)
        self.text_transformer = Transformer(input_dim=text_dim, depth=text_encode_depth, sequence_len=text_sequence_len,
                                            causal=False, heads=text_heads)
        self.to_text_latent = Linear(units=latent_dim, input_dim=text_dim, bias=False)

        assert visual_image_size % visual_patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (visual_image_size // visual_patch_size) ** 2
        patch_dim = channels * visual_patch_size ** 2

        self.visual_patch_size = visual_patch_size
        self.to_visual_embedding = Linear(units=image_dim, input_dim=patch_dim)
        self.visual_pos_embedding = Embedding(num_patches, image_dim)
        self.visual_transformer = Transformer(input_dim=image_dim, depth=visual_encode_depth, sequence_len=num_patches,
                                              causal=False, heads=visual_heads)
        self.to_visual_latent = Linear(latent_dim, image_dim, bias=False)

        self.temperature = tf.Variable(1.)

    def call(self, text, image, text_mask=None, return_loss=False):
        b, p = text.shape[0], self.visual_patch_size

        text_embeddings = self.text_embedding(text)
        text_embeddings += self.text_pos_embedding(tf.range(text.shape[1]))

        image_patches = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        image_embeddings = self.to_visual_embedding(image_patches)
        image_embeddings += self.visual_pos_embedding(tf.range(image_embeddings.shape[1]))

        encode_text = self.text_transformer(text_embeddings, mask=text_mask)
        encode_image = self.visual_transformer(image_embeddings)

        if exists(text_mask):
            text_latents = masked_mean(input_tensor=encode_text, mask=text_mask, axis=1)
        else:
            text_latents = tf.math.reduce_mean(input_tensor=encode_text, axis=1)

        image_latents = tf.math.reduce_mean(input_tensor=encode_image, axis=1)

        text_latents = self.to_text_latent(text_latents)
        image_latents = self.to_visual_latent(image_latents)

        text_latents, image_latents = map(lambda tensor: normalize(tensor, axis=-1, order=2), (text_latents, image_latents))
        temp = tf.math.exp(self.temperature)

        if not return_loss:
            sim = tf.einsum('n d, n d -> n', text_latents, image_latents) * temp
            return sim

        sim = tf.einsum('i d, j d -> i j', text_latents, image_latents) * temp
        labels = tf.range(b)
        loss = (tf.nn.softmax_cross_entropy_with_logits(labels, sim) +
                tf.nn.softmax_cross_entropy_with_logits(tf.transpose(labels), sim)) / 2
        return loss


class DALLE(Model):
    def __init__(self, *, dim, vae, num_text_tokens=10000, text_sequence_len=256, depth, heads=8, dim_head=64,
                 reversible=False, attn_dropout=0.0, ff_dropout=0.0, noncausal_attn_len=0, ignore_index=-100):
        super(DALLE, self).__init__()
        assert isinstance(vae, DiscreteVAE), 'VAE must be an instance of DiscreteVAE.'
        vae.trainable = False   # `Weights for model %s have not yet been created.` would be raised otherwise
        image_size = vae.image_size
        num_image_tokens = vae.num_tokens
        image_sequence_len = (vae.image_size // (2 ** vae.num_layers)) ** 2

        self.text_embedding = Embedding(num_text_tokens, dim)
        self.image_embedding = Embedding(num_image_tokens, dim)

        self.text_pos_embedding = Embedding(text_sequence_len + 1, dim)     # +1 for <BOS>
        self.image_pos_embedding = AxialPositionalEmbedding(dim=dim, axial_shape=(image_size, image_size))

        self.num_text_tokens = num_text_tokens  # For offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = num_image_tokens
        total_tokens = num_text_tokens + num_image_tokens
        self.total_tokens = total_tokens

        self.text_sequence_len = text_sequence_len
        self.image_sequence_len = image_sequence_len
        total_sequence_len = text_sequence_len + image_sequence_len
        self.total_sequence_len = total_sequence_len

        self.noncausal_attn_len = noncausal_attn_len

        self.vae = vae
        if exists(self.vae):
            self.vae = vae
            self.image_embedding = vae.codebook

        self.transformer = Transformer(input_dim=dim, depth=depth, sequence_len=total_sequence_len,
                                       reversible=reversible, causal=True, heads=heads, dim_head=dim_head,
                                       attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                                       noncausal_attn_len=noncausal_attn_len + 1)

        self.to_logits = Sequential([
            LayerNormalization(),
            Linear(units=self.total_tokens, input_dim=dim)
        ])

        sequence_range = tf.range(total_sequence_len)
        logits_range = tf.range(total_tokens)

        sequence_range = rearrange(sequence_range, 'n -> () n ()')
        logits_range = rearrange(logits_range, 'd -> () () d')

        self.logits_mask = tf.Variable(initial_value=(
                ((sequence_range >= text_sequence_len) & (logits_range < num_text_tokens)) |
                ((sequence_range < text_sequence_len) & (logits_range >= num_text_tokens))
        ), trainable=False, name='logits_mask')

        self.ignore_index = ignore_index

    def generate_images(self, text, *, clip=None, mask=None, filter_threshold=0.8, temperature=1):
        vae, text_sequence_len, image_sequence_len, num_text_tokens = \
            self.vae, self.text_sequence_len, self.image_sequence_len, self.num_text_tokens
        total_sequence_len = text_sequence_len + image_sequence_len

        if tf.rank(text) < 2:
            text = tf.expand_dims(input=text, axis=0)
        if exists(mask) and tf.rank(mask) < 2:
            mask = tf.expand_dims(input=mask, axis=0)

        out = text

        for cur_len in range(text.shape[1], total_sequence_len):
            is_image = cur_len >= text_sequence_len

            text, image = out[:, :text_sequence_len], out[:, text_sequence_len:]
            logits = tf.stop_gradient(self(text, image, mask=mask)[:, -1, :])
            filtered_logits = top_k(logits, threshold=filter_threshold)
            probs = tf.nn.softmax(logits=filtered_logits / temperature, axis=-1)
            sample = tf.random.categorical(logits=tf.math.log(probs), num_samples=1)
            # Offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            sample -= num_text_tokens if is_image else 0
            out = tf.concat(values=[out, sample], axis=-1)

            if out.shape[1] <= text_sequence_len:
                mask = tf.pad(tensor=mask, paddings=[[0, 0], [0, 1]], mode='CONSTANT', constant_values=True)

        text_sequence = out[:, :text_sequence_len]

        image_sequence = out[:, -image_sequence_len:]
        images = vae.decode(image_sequence)

        if exists(clip):
            scores = clip(text=text_sequence, image=images, return_loss=False)
            return images, scores

        return images

    def call(self, text, image=None, mask=None, return_loss=False):
        ignore_index, total_sequence_len = self.ignore_index, self.total_sequence_len

        # Use padding 1 as <BOS>
        text = tf.pad(tensor=text, paddings=[[0, 0], [1, 0]], mode="CONSTANT", constant_values=1)

        if exists(mask):
            mask = tf.pad(tensor=mask, paddings=[[0, 0], [1, 0]], mode="CONSTANT", constant_values=True)

        tokens = self.text_embedding(text)
        tokens += self.text_pos_embedding(tf.range(text.shape[1]))

        sequence_len = tokens.shape[1]

        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4
            if is_raw_image:
                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            image_embeddings = self.image_embedding(image)
            image_embeddings += self.image_pos_embedding(image_embeddings)
            tokens = tf.concat([tokens, image_embeddings], axis=1)
            sequence_len += image_len
            if exists(mask):
                mask = tf.pad(tensor=mask, paddings=[[0, 0], [0, image_embeddings.shape[1]]], mode='CONSTANT',
                              constant_values=True)

        # When training, if the length exceeds the total text + image length
        # Remove the last token, since it needs not to be trained
        if tokens.shape[1] > total_sequence_len:
            sequence_len -= 1
            tokens = tokens[:, :-1]

            if exists(mask):
                mask = mask[:, :-1]
        out = self.transformer(tokens, mask=mask)
        logits = self.to_logits(out)

        # Mask logits to make sure text predicts text (except last token), and image predicts image
        logits_mask = self.logits_mask[:, :sequence_len]
        max_negative_value = -tf.experimental.numpy.finfo(logits.dtype).max
        logits = tf.where(logits_mask, max_negative_value, logits)

        if not return_loss:
            return logits

        assert exists(image), 'When training, image must be supplied.'
        noncausal_attn_len = self.noncausal_attn_len
        offsetted_image = image + self.num_text_tokens
        offsetted_image = tf.cast(offsetted_image, dtype=text.dtype)
        labels = tf.concat(values=[text[:, 1:], offsetted_image], axis=1)

        if noncausal_attn_len:
            sequence_range = tf.range(sequence_len)
            noncausal_attn_mask = sequence_range < noncausal_attn_len
            noncausal_attn_mask = rearrange(noncausal_attn_mask, 'n -> () n')
            labels = tf.where(noncausal_attn_mask, ignore_index, labels)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        return loss
