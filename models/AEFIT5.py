
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

import tensorflow as tf
import abc

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import ipysh
import models
import models.layers


"""
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""
class AEFIT5(models.base.VAE):
    ''' General Autoencoder Fit Model for TF 2.0
    '''    
    def __init__(self, feature_dim=40, latent_dim=2, dprate=0., activation=tf.nn.relu, beta=1., 
                 geometry=[20,20,10], scale=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim        
        self.dprate = dprate
        self.scale = scale
        self.activation = activation
        self.beta = tf.Variable(beta, dtype=tf.float32, name='beta', trainable=False)
        self.apply_sigmoid = False
        self.bypass = False
        self.akl = True
        self.set_model(feature_dim, latent_dim, 
                            dprate=dprate,
                            scale=scale, 
                            activation=activation,
                            geometry=geometry)

        self.output_names = self.generative_net.output_names
        
        self.akl_loss = 0.
        self.mkl_loss = 0.
        def akl(x,y): return self.akl_loss
        def mkl(x,y): return self.mkl_loss 
        def b(x,y): return self.beta
        def mse(x,y): return tf.reduce_mean(self.loss(x,y))
        self.compile(
            #optimizer  = tf.keras.optimizers.SGD(learning_rate=1e-3),
            #optimizer  = tf.keras.optimizers.Adagrad(learning_rate=1e-3),
            optimizer   = tf.keras.optimizers.Adam(learning_rate=1e-3),            
            loss        = self.compute_mse_loss,
            # loss       = self.compute_cross_entropy_loss,
            # logit_loss = True,
            metrics     = [mse,akl,mkl,b],
            run_eagerly = True
        )
        print('AEFIT5 ready:')

    
    def set_model(self, feature_dim, latent_dim, dprate=0., activation=tf.nn.relu, 
                  geometry=[20,20,10], scale=1):

        class LsInitializer(tf.keras.initializers.Initializer):
            """Initializer for latent layer"""
            def __init__(self, axis=1):
                super(LsInitializer, self).__init__()
                self.axis = axis

            def __call__(self, shape, dtype=tf.dtypes.float32):
                dtype = tf.dtypes.as_dtype(dtype)
                if not dtype.is_numpy_compatible or dtype == tf.dtypes.string:
                    raise ValueError("Expected numeric or boolean dtype, got %s." % dtype)
                axis = self.axis
                shape[axis] = int(shape[axis]/2)
                identity = tf.initializers.identity()(shape)
                return tf.concat([identity, tf.zeros(shape)], axis=axis)

        def add_dense_encode(self, fdim=feature_dim, ldim=latent_dim, geometry=[20,20,10,10]):
            for _,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale, activation=activation))
                self.add(tf.keras.layers.Dropout(dprate))
            if len(geometry) == 0: initializer = LsInitializer()
            else : initializer = None            
            self.add(tf.keras.layers.Dense(ldim, activation='linear', use_bias=False, kernel_initializer=initializer))
            return self

        def add_dense_decode(self, fdim=feature_dim, ldim=latent_dim, geometry=[10,10,20,20]):            
            for _,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale, activation=activation))
                self.add(tf.keras.layers.Dropout(dprate))
            if len(geometry) == 0: initializer = tf.initializers.identity()
            else : initializer = None
            self.add(tf.keras.layers.Dense(fdim, activation='linear', use_bias=False, kernel_initializer=initializer))            
            return self
        # add methods to Sequential class
        tf.keras.Sequential.add_dense_encode = add_dense_encode
        tf.keras.Sequential.add_dense_decode = add_dense_decode
        
        self.inference_rlv  = models.layers.Relevance1D(name=self.name+'_iRlv', activation='linear', kernel_initializer=tf.initializers.ones)
        self.generative_rlv = models.layers.Relevance1D(name=self.name+'_gRlv', activation='linear', kernel_initializer=tf.initializers.ones)

        ## INFERENCE ##
        inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim,)),
            tf.keras.layers.Lambda(lambda x: tf.where(tf.math.is_nan(x),tf.zeros_like(x),x)), 
            # models.layers.NaNDense(feature_dim),
            # models.layers.Relevance1D(name=self.name+'_iRlv', activation='linear', kernel_initializer=tf.initializers.ones),
            self.inference_rlv,
        ]).add_dense_encode(ldim=2*latent_dim, geometry=geometry)

        ## GENERATION ##
        generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),
            # models.layers.Relevance1D(name=self.name+'_gRlv', activation='linear', kernel_initializer=tf.initializers.ones),
            self.generative_rlv,
        ]).add_dense_decode(geometry=geometry[::-1])
        
        self.inference_net = inference_net
        self.generative_net = generative_net

        return inference_net, generative_net    

    def set_rlv_bypass(self,value = True):
        self.inference_rlv.set_bypass(value)
        self.generative_rlv.set_bypass(value)

    def reparametrize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
    def encode(self, X, training=None):
        mean, logvar = tf.split(self.inference_net(X, training=training), num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, s, training=True, apply_sigmoid=None):
        x = self.generative_net(s, training=training)
        if apply_sigmoid is None: apply_sigmoid = self.apply_sigmoid        
        if apply_sigmoid is True and training is False:
            x = tf.sigmoid(x)
        return x

    def call(self, xy, training=True):
        def vae_logN_pdf(sample, mean, logvar):
            log2pi = tf.math.log(2. * np.pi)
            return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)

        att = tf.math.is_nan(xy)
        xy  = tf.where(att, tf.zeros_like(xy), xy)
        mean, logvar = self.encode(xy, training=training)
        z = self.reparametrize(mean,logvar)
        XY = self.decode(z, training=training)
        akl_loss = -0.5 * tf.reduce_sum(1. + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
        mkl_loss = tf.reduce_sum(vae_logN_pdf(z, mean, logvar) - vae_logN_pdf(z, 0., 1.), axis=1)
        if training is not False:
            XY  = tf.where(att, tf.zeros_like(XY), XY)
        if self.bypass:
            XY = 0.*XY + xy # add dummy gradients passing through the ops
            akl_loss = 0.
            mkl_loss = 0.
        
        self.akl_loss = akl_loss
        self.mkl_loss = mkl_loss
        if self.akl is True:
            kl_loss = self.beta * tf.reduce_mean(akl_loss)
        else:
            kl_loss = self.beta * tf.reduce_mean(mkl_loss)
        if training is True:
            self.add_loss(lambda: kl_loss)
        # else:
        #     self.add_loss(0.)
        return XY

    def train_step(self, data, training=True):
        xy = data[0]
        with tf.GradientTape() as tape:
            XY = self.call(xy, training=training)
            loss = self.loss(xy, XY) # tf.reduce_mean( self.loss(xy, XY) + self.losses[0] )

        if training:
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss":loss}

    def compile(self, optimizer=None, loss=None, logit_loss=False, metrics=None, **kwargs):
        if optimizer is None: 
            if self.optimizer: optimizer = self.optimizer
            else             : optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        if loss is None or (hasattr(self, 'loss') and loss == self.loss): loss_wrapper = self.loss
        else: 
            self.apply_sigmoid = logit_loss
            loss_wrapper = lambda xy,XY: loss( tf.where(tf.math.is_nan(xy), tf.zeros_like(xy), xy), 
                                               tf.where(tf.math.is_nan(xy), tf.zeros_like(XY), XY))
            # loss_wrapper = self.vae_loss(loss)
        return super().compile(optimizer, loss=loss_wrapper, metrics=metrics, **kwargs)
    
    def compute_cross_entropy_loss(self, xy, XY):
        crossen =  tf.nn.sigmoid_cross_entropy_with_logits(logits=XY, labels=xy)
        logpx_z =  tf.reduce_mean( crossen , axis=1)
        return logpx_z

    def compute_mse_loss(self, xy, XY):
        return tf.keras.losses.mse(y_pred=XY, y_true=xy)

    def recover(self,x):
        xr = self.call(x, training=False)
        return tf.where(tf.math.is_nan(x),xr,x)







    