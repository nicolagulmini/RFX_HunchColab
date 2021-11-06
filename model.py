# VAE

# note that it is necessary to compile it before training it!

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
           
class betaScheduler(keras.callbacks.Callback):

    def __init__(self, schedule):
        super(betaScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        beta = float(tf.keras.backend.get_value(self.model.beta))
        scheduled_beta = self.schedule(epoch, beta)
        tf.keras.backend.set_value(self.model.beta, scheduled_beta)
        print("Epoch %d: beta parameter is %f." % (epoch, scheduled_beta))

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
    def __init__(self, input_shape, latent_dim, beta=1., **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder(input_shape, latent_dim)
        self.decoder = decoder(latent_dim, input_shape)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta = tf.Variable(beta, trainable=False)

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
            reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction="auto")(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta*kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def reconstruct_input(self, data, mean=True):
        z_mean, z_log_var, z = self.encoder(data)
        if mean:
            reconstruction = self.decoder(z_mean)
        else:
            reconstruction = self.decoder(z)
        reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction="auto")(data, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.beta*kl_loss
        return reconstruction, reconstruction_loss, kl_loss, total_loss
    
    def get_lr(self):
        return self.optimizer.lr.numpy()

    def plot_latent_space(self, sub_len=20, step=.2, savefig=False): # generalizza a più dimensioni
        fig, axs = plt.subplots(sub_len, sub_len, sharex='all', sharey='all', figsize=(20, 20))
        row_index = 0
        for first_dim in range(-int(sub_len/2), int(sub_len/2)):
            col_index = 0
            for second_dim in range(-int(sub_len/2), int(sub_len/2)):
                point = np.array([[first_dim*step, second_dim*step]])
                generated = self.decoder.predict(point).tolist()[0]
                x_axis = [el[0] for el in generated]
                y_axis = [el[1] for el in generated]
                axs[row_index][col_index].plot(x_axis, y_axis, linewidth=1)
                axs[row_index][col_index].get_xaxis().set_visible(False)
                axs[row_index][col_index].get_yaxis().set_visible(False)
                col_index += 1
            row_index += 1
        if savefig == True:
            plt.savefig('latent_space.png')
