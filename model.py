import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from math import pi
from tensorflow import keras
from tensorflow.keras import layers
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame

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
    masked_input = tf.keras.layers.Lambda(lambda x: tf.where(tf.math.is_nan(x), tf.zeros_like(x), x))(encoder_inputs)
    flat = layers.Flatten()(masked_input)
    dense = layers.Dense((200), activation='relu')(flat)
    drop = layers.Dropout(0.1)(dense)
    dense = layers.Dense((200), activation='relu')(drop)
    drop = layers.Dropout(0.2)(dense)
    dense = layers.Dense((100), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)
    dense = layers.Dense((100), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)
    
    z_mean = layers.Dense((latent_dim), use_bias=False, name="z_mean")(drop)
    z_log_var = layers.Dense((latent_dim), name="z_log_var")(drop)
    z = Sampling()([z_mean, z_log_var])

    encoder_model = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name=name)

    if summary:
        encoder_model.summary()

    return encoder_model

def decoder(latent_dim, target_shape, name='decoder', summary=False):
    latent_inputs = keras.Input(shape=(latent_dim,)) # this layer takes only the sampled z vector
    
    dense = layers.Dense((100), activation='relu')(latent_inputs) # 400
    drop = layers.Dropout(0.1)(dense)
    dense = layers.Dense((100), activation='relu')(drop) 
    drop = layers.Dropout(0.2)(dense)
    dense = layers.Dense((200), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)
    dense = layers.Dense((200), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)
    
    dense = layers.Dense((np.prod(target_shape)), use_bias=False, activation='sigmoid')(drop)
    reshaped = layers.Reshape(target_shape)(dense)
    decoder_model = keras.Model(latent_inputs, reshaped, name=name)

    if summary:
        decoder_model.summary()

    return decoder_model

class VAE(keras.Model):
    def __init__(self, input_shape, latent_dim=2, beta=1., **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
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
            masked_data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
            masked_reconstruction = tf.where(tf.math.is_nan(data), tf.zeros_like(data), reconstruction)
            reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction="auto")(masked_data, masked_reconstruction)
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
        data = tf.cast(data, dtype=tf.float32)
        z_mean, z_log_var, z = self.encoder(data)
        if mean:
            reconstruction = self.decoder(z_mean)
        else:
            reconstruction = self.decoder(z)
        masked_data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
        mask = tf.where(tf.math.is_nan(data), tf.zeros_like(data), tf.ones_like(data))
        masked_reconstruction = tf.where(tf.math.is_nan(data), tf.zeros_like(reconstruction), reconstruction)
        reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction="auto")(masked_data, masked_reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + self.beta*kl_loss
        return reconstruction, reconstruction_loss, kl_loss, total_loss
    
    def get_lr(self):
        return self.optimizer.lr.numpy()

    def plot_latent_space(self, sub_len=20, step=.2, dimensions=[0, 1], savefig=False, name='latent_space.png'): 
        fig, axs = plt.subplots(sub_len, sub_len, sharex='all', sharey='all', figsize=(sub_len, sub_len))

        if (not len(dimensions) == 2) or (dimensions[0] >= self.latent_dim) or (dimensions[1] >= self.latent_dim):
            print("Error in dimensions. Return.")
            return
        first_pos, second_pos = dimensions
                       
        row_index = 0
        for first_dim in range(-int(sub_len/2), int(sub_len/2)):
            col_index = 0
            for second_dim in range(-int(sub_len/2), int(sub_len/2)):
                tmp_vec = [0 for _ in range(self.latent_dim)]
                tmp_vec[first_pos] = first_dim*step
                tmp_vec[second_pos] = second_dim*step
                point = np.array([tmp_vec])
                generated = self.decoder.predict(point).tolist()[0]
                x_axis = [el[0] for el in generated]
                y_axis = [el[1] for el in generated]
                axs[row_index][col_index].plot(x_axis, y_axis, linewidth=1)
                axs[row_index][col_index].get_xaxis().set_visible(False)
                axs[row_index][col_index].get_yaxis().set_visible(False)
                col_index += 1
            row_index += 1
        if savefig == True:
            plt.savefig(name)
            
    def parallel_latent_visualization(self, data, favourite_dim=0, colorscale='Electric', showscale=True):
        # https://plotly.com/python/parallel-coordinates-plot/
        mean = self.encoder(data)[0].numpy()
        dimensions = []
        for _ in range(self.latent_dim):
            dimensions.append(np.array([el[_] for el in mean]))

        fig = go.Figure(data = go.Parcoords(
                line = dict(color = dimensions[favourite_dim], colorscale=colorscale,
                        showscale=showscale,
                        cmin = min(dimensions[favourite_dim]),
                        cmax = max(dimensions[favourite_dim])),
                dimensions = list([
                    dict(range = [min(dimensions[i]),max(dimensions[i])],
                        label = str(i+1) + " - dimension", values = dimensions[i]) for i in range(self.latent_dim)])))
        fig.show()
