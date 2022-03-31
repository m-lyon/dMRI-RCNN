'''Model layers'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.keras.utils import tf_utils  # pylint: disable=no-name-in-module


class DistributedConv3D(layers.Layer):
    '''TimeDistributed Conv3D'''

    def __init__(
        self,
        filters,
        kernel_size,
        instance_norm=False,
        batch_norm=False,
        name='',
        activation='swish',
    ):
        '''TimeDistributed Conv3D'''
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.instance_norm = instance_norm
        self.batch_norm = batch_norm
        self.activation = activation

        # Initialise layers
        self.conv_layer = self._get_conv_layer()
        self.activation_layer = self._get_activation_layer()
        self.norm_layer = self._get_norm_layer()

    def _get_activation_layer(self):
        act_layers = {
            'swish': layers.Activation(keras.activations.swish),
            'relu': layers.ReLU(),
        }
        assert (
            self.activation in act_layers
        ), f'{self.activation} not in allowed activations.'

        return layers.TimeDistributed(
            act_layers[self.activation], name=f'{self.name}_{self.activation}'
        )

    def _get_conv_layer(self):
        return layers.TimeDistributed(
            layers.Conv3D(
                self.filters,
                self.kernel_size,
                padding='same',
                kernel_initializer=keras.initializers.he_uniform(),
            ),
            name=f'{self.name}_conv',
        )

    def _get_norm_layer(self):
        if self.instance_norm:
            return layers.LayerNormalization(axis=(-2, -3, -4))
        if self.batch_norm:
            return layers.BatchNormalization()
        return None

    def call(self, inputs, training=False):
        # pylint: disable=arguments-differ
        # Convolution
        tensor = self.conv_layer(inputs, training=training)
        # Activation
        if self.activation_layer is not None:
            tensor = self.activation_layer(tensor, training=training)
        # Normalisation
        if self.norm_layer is not None:
            tensor = self.norm_layer(tensor, training=training)

        return tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'batch_norm': self.batch_norm,
                'instance_norm': self.instance_norm,
                'activation': self.activation,
            }
        )
        return config

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            raise AttributeError('the input shape passed is a list')
        return self.conv_layer.compute_output_shape(input_shape)


class RepeatBVector(layers.Layer):
    '''Expands and tiles bvec tensor to spatial dimensions
        Must be at least rank 4 tensor

    Call arguments:
        bvec (tf.Tensor): A rank 3 tensor,
                (batch, time, 3)
            spatial dimensions are inserted at dimension
            position -2.
        spatial_dims (Tuple[int,]): Spatial dimensions
            to expand and tile bvec into.

    Output shape:
        (batch, time, spatial_dims, 3)
    '''

    def call(self, bvec, spatial_dims):
        # pylint: disable=arguments-differ

        multiples = tf.constant([1, 1] + list(spatial_dims) + [1], tf.int32)
        for _ in spatial_dims:
            bvec = tf.expand_dims(bvec, axis=-2)
        bvec = tf.tile(bvec, multiples)

        return bvec


class RepeatTensor(layers.Layer):
    '''Extends and repeats an arbitrary tensor

    Call arguments:
        tensor (tf.Tensor): Some tensor to repeat
        axis (int): Axis index where repeats
            will be inserted.
        ref_tensor (tf.Tensor): A tensor where
            where the size of axis number `ref_axis`
            is the number of times to repeat `tensor`.
            e.g. repeats = tf.shape(`ref_tensor`)[`ref_axis`]
        ref_axis (int): See above.
    '''

    def call(self, tensor, axis, ref_tensor, ref_axis):
        # pylint: disable=arguments-differ
        repeats = tf.shape(ref_tensor)[ref_axis]
        tensor = tf.expand_dims(tensor, axis=axis)
        tensor = tf.repeat(tensor, repeats, axis=axis)

        return tensor
