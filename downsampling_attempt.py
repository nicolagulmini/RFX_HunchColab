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
           
def encoder(input_shape, latent_dim, name='encoder', summary=False):
    encoder_inputs = keras.Input(shape=input_shape)
    flat = layers.Flatten()(encoder_inputs)

    dense = layers.Dense((20*latent_dim), activation='relu')(flat)
    drop = layers.Dropout(0.1)(dense)
    dense = layers.Dense((20*latent_dim), activation='relu')(drop)
    drop = layers.Dropout(0.2)(dense)
    dense = layers.Dense((10*latent_dim), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)
    dense = layers.Dense((10*latent_dim), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)
    
    z_mean = layers.Dense((latent_dim), use_bias=False, name="z_mean")(drop)
    z_log_var = layers.Dense((latent_dim), name="z_log_var")(dense)
    z = Sampling()([z_mean, z_log_var])

    encoder_model = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name=name)

    if summary:
        encoder_model.summary()

    return encoder_model

def decoder(latent_dim, low_res_shape, high_res_shape, window_size=10, window_slide=10, name='decoder', summary=False):
    latent_inputs = keras.Input(shape=(latent_dim,)) # this layer takes only the sampled z vector
    
    dense = layers.Dense((10*latent_dim), activation='relu')(latent_inputs) 
    drop = layers.Dropout(0.1)(dense)
    dense = layers.Dense((10*latent_dim), activation='relu')(drop) 
    drop = layers.Dropout(0.2)(dense)
    dense = layers.Dense((20*latent_dim), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)
    dense = layers.Dense((20*latent_dim), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)

    # larger
    dense = layers.Dense((200*latent_dim), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)
    dense = layers.Dense((200*latent_dim), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)
    dense = layers.Dense((200*latent_dim), activation='relu')(drop)
    drop = layers.Dropout(0.3)(dense)
    
    dense = layers.Dense((np.prod(high_res_shape)), use_bias=False, activation='sigmoid')(drop)
    high_res_output = layers.Reshape(high_res_shape)(dense)
    low_res_output = layers.MaxPooling1D(pool_size=window_size, strides=window_slide, padding='valid')(high_res_output) # modify pool size e strides
    decoder_model = keras.Model(latent_inputs, [high_res_output, low_res_output], name=name)

    if summary:
        decoder_model.summary()

    return decoder_model

def down_sampler(input_shape, output_shape):
    input = keras.Input(input_shape) 
    flat = layers.Flatten()(input)
    dense = layers.Dense((np.prod(output_shape)))(flat) 
    out = layers.Reshape(output_shape)(dense)
    model = keras.Model(input, out)
    return model

class VAE(keras.Model):
    def __init__(self, low_res_shape, high_res_shape, window_size=10, window_slide=10, latent_dim=2, beta=1., **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = encoder(low_res_shape, latent_dim)
        self.decoder = decoder(latent_dim, low_res_shape, high_res_shape, window_size, window_slide)
        self.down_sampler = down_sampler(high_res_shape, low_res_shape)
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
            _, reconstruction = self.decoder(z)
            learning_downsampling = self.down_sampler(_)
            reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction="auto")(data, learning_downsampling)
            between_architectures_loss = tf.keras.losses.MeanSquaredError(reduction="auto")(reconstruction, learning_downsampling)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.beta*kl_loss + between_architectures_loss
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
    
    def reconstruct_input(self, data, mean=True, reshaped=False, downsampler=False):
        z_mean, z_log_var, z = self.encoder(data)
        index = 0
        if reshaped:
            index = 1
        if mean:
            reconstructions = self.decoder(z_mean)
        else:
            reconstructions = self.decoder(z)
        if downsampler:
            recon = self.down_sampler(reconstructions[0])
        else:
            recon = reconstructions[index]
        return recon
    
    def get_lr(self):
        return self.optimizer.lr.numpy()

    def plot_latent_space(self, sub_len=20, step=.2, dimensions=[0, 1], savefig=False, name='latent_space.png'): 
        fig, axs = plt.subplots(sub_len, sub_len, sharex='all', sharey='all', figsize=(sub_len, sub_len))

        first_pos, second_pos = dimensions
                       
        row_index = 0
        for first_dim in range(-int(sub_len/2), int(sub_len/2)):
            col_index = 0
            for second_dim in range(-int(sub_len/2), int(sub_len/2)):
                tmp_vec = [0 for _ in range(self.latent_dim)]
                tmp_vec[first_pos] = first_dim*step
                tmp_vec[second_pos] = second_dim*step
                point = np.array([tmp_vec])
                generated = self.decoder.predict(point)[1].tolist()[0]
                x_axis = [el[0] for el in generated]
                y_axis = [el[1] for el in generated]
                axs[row_index][col_index].plot(x_axis, y_axis, linewidth=1)
                axs[row_index][col_index].get_xaxis().set_visible(False)
                axs[row_index][col_index].get_yaxis().set_visible(False)
                col_index += 1
            row_index += 1
        if savefig == True:
            plt.savefig(name)
