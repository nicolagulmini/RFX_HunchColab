from __future__ import print_function

import numpy as np
import tensorflow as tf
import abc

"""
.##....##....###....##....##....########..########.##....##..######..########
.###...##...##.##...###...##....##.....##.##.......###...##.##....##.##......
.####..##..##...##..####..##....##.....##.##.......####..##.##.......##......
.##.##.##.##.....##.##.##.##....##.....##.######...##.##.##..######..######..
.##..####.#########.##..####....##.....##.##.......##..####.......##.##......
.##...###.##.....##.##...###....##.....##.##.......##...###.##....##.##......
.##....##.##.....##.##....##....########..########.##....##..######..########
"""

class NaNDense(tf.keras.layers.Dense):
    """Just your regular densely-connected NN layer.
    """
    def __init__(self,
               units,
               activation=None,
               use_bias=True,
               **kwargs):
        super(NaNDense, self).__init__( units, activation, **kwargs)
        
    
    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        outputs = tf.matmul(inputs, self.kernel)
        if self.use_bias: 
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs



class Reparametrize1D(tf.keras.layers.Layer):
    """ VAE REPARAMETRIZATION LAYER
    """
    def __init__(self, **kwargs):
        super(Reparametrize1D, self).__init__(**kwargs)
    
    @tf.function
    def reparametrize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    @tf.function
    def call(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        mean, logvar = tf.split(inputs, num_or_size_splits=2, axis=1)
        akl_loss = -0.5 * tf.reduce_sum(1. + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
        mean = self.reparametrize(mean,logvar)
        self.add_loss( akl_loss, inputs=True )
        return mean
        
    

class RelUnitNorm(tf.keras.constraints.Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.
    """
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        # w =  w / ( tf.keras.backend.epsilon() + 
        #      tf.sqrt( tf.reduce_sum(tf.square(w), axis=self.axis, keepdims=True)))
        w =  w / ( tf.keras.backend.epsilon() + 
             tf.sqrt( tf.reduce_max(tf.square(w), axis=self.axis, keepdims=True)))
        return w

    def get_config(self):
        return {'axis': self.axis}


class Relevance1D(tf.keras.layers.Dropout):
    def __init__(self,
                activation=None,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=None,
                kernel_constraint=RelUnitNorm(),
                **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Relevance1D, self).__init__( 0., **kwargs )
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)        
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bypass = False
        self.supports_masking = True
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=1)

    def build(self, input_shape):
        dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx)
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                            'should be defined. Found `None`.')
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=1, axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,            
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
            
        self.built = True

    def call(self, inputs):
        inputs  = tf.convert_to_tensor(inputs)
        # inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        # outputs = tf.cond(self.bypass is True, lambda: tf.multiply( inputs , self.kernel), lambda: inputs )
        if self.bypass is True: outputs = inputs
        else: outputs = tf.multiply( inputs , self.kernel )
        # if self.activation is not None:
        #     return self.activation(outputs)  # pylint: disable=not-callable
        outputs = super(Relevance1D, self).call(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError( 'The innermost dimension of input_shape must be defined, but saw: %s' % input_shape)
        return input_shape

    def set_bypass(self, value=True):
        self.bypass = value

    def get_config(self):
        # config = {
        #     'activation': tf.keras.activations.serialize(self.activation),
        #     'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
        #     'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
        # }
        base_config = super(tf.keras.layers.Dropout, self).get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return dict(list(base_config.items()))
