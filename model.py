# VAE

# note that it is necessary to compile it before training it!

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def encoder(input_shape, latent_dim, name='encoder', summary=False):
    encoder_inputs = keras.Input(shape=input_shape)
    flat = layers.Flatten()(encoder_inputs)

    dense = layers.Dense((20*latent_dim))(flat) # no activation
    dense = layers.Dense((20*latent_dim))(dense) # no activation
    dense = layers.Dense((10*latent_dim))(dense) # no activation
    dense = layers.Dense((10*latent_dim))(dense) # no activation
    
    z_mean = layers.Dense((latent_dim), use_bias=False, name="z_mean")(dense) # also here a linear activation function 
    z_log_var = layers.Dense((latent_dim), name="z_log_var")(dense)
    z = Sampling()([z_mean, z_log_var])

    encoder_model = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name=name)

    if summary:
        encoder_model.summary()

    return encoder_model

def decoder(latent_dim, target_shape, name='decoder', summary=False):
    latent_inputs = keras.Input(shape=(latent_dim,)) # this layer takes only the sampled z vector
    
    dense = layers.Dense((10*latent_dim))(latent_inputs) # no activation
    dense = layers.Dense((10*latent_dim))(dense) # no activation
    dense = layers.Dense((20*latent_dim))(dense) # no activation
    dense = layers.Dense((20*latent_dim))(dense) # no activation
    
    dense = layers.Dense((np.prod(target_shape)), activation='sigmoid')(dense) # no activation
    reshaped = layers.Reshape(target_shape)(dense)
    decoder_model = keras.Model(latent_inputs, reshaped, name=name)

    if summary:
        decoder_model.summary()

    return decoder_model

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta = 0.001

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta*kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def reconstruct_input(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z_mean)
        reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")(data, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.beta*kl_loss
        return reconstruction, reconstruction_loss, kl_loss, total_loss
    
    def learning_rate_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        return lr*tf.math.exp(-0.1)
