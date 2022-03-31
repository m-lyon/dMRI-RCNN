'''Autoencoder implementation'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .layers import RepeatBVector, RepeatTensor, DistributedConv3D


def get_adam_opt():
    '''Returns Adam optimizer, initialized.'''

    adam = tf.keras.optimizers.Adam(
        learning_rate=tf.Variable(0.001),
        beta_1=tf.Variable(0.9),
        beta_2=tf.Variable(0.999),
        epsilon=tf.Variable(1e-7),
    )
    _ = adam.iterations
    adam.decay = tf.Variable(0.0)

    return adam


def get_3d_encoder(**kwargs):
    '''Genereates the encoder

    Keyword Args:
        q_in (int): Number of input q samples in model, set as `None` to
            allow varying number of timesteps. Default: `None`
        in_vox_shape (Tuple[int,int,int]): voxel input shape
            Default: (10, 10, 10)
        lstm_size (int): Number of units within the 3DConvLSTM.
            Default: 48

    Returns:
        encoder (tf.keras.models.Model): Encoder model.
    '''
    # pylint: disable=no-member
    q_in = kwargs.setdefault('q_in', None)
    in_vox_shape = kwargs.setdefault('in_vox_shape', (10, 10, 10))
    lstm_size = kwargs.setdefault('lstm_size', 48)

    # Inputs
    input_img = keras.Input(shape=(q_in,) + in_vox_shape, name='image')
    input_vec = keras.Input(shape=(q_in, 3), name='bvec')

    # Add channel dimension to input dMRI images
    img_tensor = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_img)
    # Duplicate B-vector so each pixel has B-vector
    bvec_tensor = RepeatBVector()(input_vec, in_vox_shape)
    # Concatenate B-Vector with input image
    qspace_tensor = layers.Concatenate()([img_tensor, bvec_tensor])

    # Initial B-vector convolution
    init_tensor = DistributedConv3D(200, 1, name='init', instance_norm=True)(
        qspace_tensor
    )

    # Subsequent convolutional layers to extract features
    conv11_tensor = DistributedConv3D(104, 1, name='ecv11', instance_norm=True)(
        init_tensor
    )
    conv12_tensor = DistributedConv3D(200, 2, name='ecv12', instance_norm=True)(
        init_tensor
    )
    conv13_tensor = DistributedConv3D(72, 3, name='ecv13', instance_norm=True)(
        init_tensor
    )
    conv1_tensor = layers.Concatenate()(
        [conv11_tensor, conv12_tensor, conv13_tensor, qspace_tensor]
    )

    # Second convolutional block
    conv21_tensor = DistributedConv3D(280, 1, name='ecv21', batch_norm=True)(
        conv1_tensor
    )
    conv22_tensor = DistributedConv3D(240, 2, name='ecv22', batch_norm=True)(
        conv1_tensor
    )
    conv23_tensor = DistributedConv3D(144, 3, name='ecv23', batch_norm=True)(
        conv1_tensor
    )
    conv2_tensor = layers.Concatenate()(
        [conv21_tensor, conv22_tensor, conv23_tensor, qspace_tensor]
    )

    # Compress to latent space
    latent_tensor = DistributedConv3D(32, 1, name='ecvl1', batch_norm=True)(
        conv2_tensor
    )
    latent_tensor = DistributedConv3D(88, 1, name='ecvl2', batch_norm=True)(
        latent_tensor
    )

    out_tensor = layers.ConvLSTM3D(lstm_size, 1, name='conv_lstm')(latent_tensor)

    encoder = keras.models.Model(
        inputs=[input_img, input_vec], outputs=out_tensor, name='encoder'
    )

    return encoder


def get_3d_decoder(**kwargs):
    '''Genereates the decoder

    Keyword Args:
        q_out (int): Number of output q samples in model, set as `None` to
            allow varying number of timesteps. Default: `None`.
        in_vox_shape (Tuple[int,int,int]): voxel input shape
            Default: (10, 10, 10)
        lstm_size (int): Number of units within the 3DConvLSTM.
            Default: 48

    Returns:
        decoder (tf.keras.models.Model): Decoder model.
    '''
    q_out = kwargs.setdefault('q_out', None)
    in_vox_shape = kwargs.setdefault('in_vox_shape', (10, 10, 10))
    lstm_size = kwargs.setdefault('lstm_size', 48)
    latent_shape = in_vox_shape + (lstm_size,)

    # Inputs
    input_state = keras.Input(shape=(latent_shape), name='state')
    input_vecs = keras.Input(shape=(q_out, 3), name='bvec')

    # Decoder
    state_tensor = RepeatTensor()(input_state, 1, input_vecs, 1)
    bvec_tensor = RepeatBVector()(input_vecs, in_vox_shape)
    latent_tensor = layers.Concatenate()([state_tensor, bvec_tensor])

    # Initial B-vector convolution
    latent_tensor = DistributedConv3D(176, 1, name='dcvl1', batch_norm=True)(
        latent_tensor
    )
    latent_tensor = DistributedConv3D(224, 1, name='dcvl2', batch_norm=True)(
        latent_tensor
    )
    latent_tensor = layers.Concatenate()([latent_tensor, bvec_tensor])

    # Subsequent convolutional layers to extract features
    conv11_tensor = DistributedConv3D(240, 1, name='dcv11', batch_norm=True)(
        latent_tensor
    )
    conv12_tensor = DistributedConv3D(256, 2, name='dcv12', batch_norm=True)(
        latent_tensor
    )
    conv13_tensor = DistributedConv3D(136, 3, name='dcv13', batch_norm=True)(
        latent_tensor
    )
    conv1_tensor = layers.Concatenate()(
        [conv11_tensor, conv12_tensor, conv13_tensor, bvec_tensor]
    )

    # Second convolutional block
    conv21_tensor = DistributedConv3D(176, 1, name='dcv21', batch_norm=True)(
        conv1_tensor
    )
    conv22_tensor = DistributedConv3D(136, 2, name='dcv22', batch_norm=True)(
        conv1_tensor
    )
    conv23_tensor = DistributedConv3D(88, 3, name='dcv23', batch_norm=True)(
        conv1_tensor
    )
    conv2_tensor = layers.Concatenate()(
        [conv21_tensor, conv22_tensor, conv23_tensor, bvec_tensor]
    )

    conv3_tensor = DistributedConv3D(16, 1, name='dcv3')(conv2_tensor)
    out_tensor = DistributedConv3D(1, 1, name='dc4', activation='relu')(conv3_tensor)

    out_tensor = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(out_tensor)

    # Create decoder model instance
    decoder = keras.models.Model(
        inputs=[input_state, input_vecs], outputs=out_tensor, name='decoder'
    )

    return decoder


def get_3d_autoencoder(weights=None, **kwargs):
    '''Generates Autoencoder Model

    Args:
        weights (str): Path to saved weights, leave as `None`
            to omit loading weights.

    Keyword Args:
        q_in (int): Number of input q samples in model, set as `None` to
            allow varying number of timesteps. Default: `None`.
        q_out (int): Number of output q samples in model, set as `None` to
            allow varying number of timesteps. Default: `None`.
        in_vox_shape (Tuple[int,int,int]): voxel input shape
            Default: (10, 10, 10)
        lstm_size (int): Number of units within the 3DConvLSTM.
            Default: 48
        loss (Union[str, tf.keras.losses.Loss]): Loss used in model training.
            see https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
            for example inputs. Default: "mae"

    Returns:
        autoencoder (tf.keras.models.Model): Autoencoder model.
    '''
    q_in = kwargs.setdefault('q_in', None)
    q_out = kwargs.setdefault('q_out', None)
    in_vox_shape = kwargs.setdefault('in_vox_shape', (10, 10, 10))
    kwargs.setdefault('lstm_size', 48)
    loss = kwargs.get('loss', 'mae')

    # Instantiate encoder & decoder models
    encoder = get_3d_encoder(**kwargs)
    decoder = get_3d_decoder(**kwargs)

    # Define inputs
    encoder_imgs = keras.Input(shape=(q_in,) + in_vox_shape, name='enc_images')
    encoder_vecs = keras.Input(shape=(q_in, 3), name='enc_bvecs')
    decoder_vecs = keras.Input(shape=(q_out, 3), name='dec_bvecs')

    # Connect encoder and decoder
    state = encoder([encoder_imgs, encoder_vecs])
    out_imgs = decoder([state, decoder_vecs])

    autoencoder = keras.models.Model(
        inputs=[encoder_imgs, encoder_vecs, decoder_vecs],
        outputs=out_imgs,
        name='autoencoder',
    )

    autoencoder.compile(loss=loss, optimizer=get_adam_opt())

    if weights is not None:
        autoencoder.load_weights(weights).assert_consumed()

    return autoencoder


def get_1d_encoder(**kwargs):
    '''Genereates the encoder. This is 1D in the sense that all operations are pointwise.

    Keyword Args:
        q_in (int): Number of input q samples in model, set as `None` to
            allow varying number of timesteps. Default: `None`
        in_vox_shape (Tuple[int,int,int]): voxel input shape
            Default: (10, 10, 10)
        lstm_size (int): Number of units within the 3DConvLSTM.
            Default: 48

    Returns:
        encoder (tf.keras.models.Model): Encoder model.
    '''
    # pylint: disable=no-member
    q_in = kwargs.setdefault('q_in', None)
    in_vox_shape = kwargs.setdefault('in_vox_shape', (10, 10, 10))
    lstm_size = kwargs.setdefault('lstm_size', 48)

    # Inputs
    input_img = keras.Input(shape=(q_in,) + in_vox_shape, name='image')
    input_vec = keras.Input(shape=(q_in, 3), name='bvec')

    # Add channel dimension to input dMRI images
    img_tensor = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_img)
    # Duplicate B-vector so each pixel has B-vector
    bvec_tensor = RepeatBVector()(input_vec, in_vox_shape)
    # Concatenate B-Vector with input image
    qspace_tensor = layers.Concatenate()([img_tensor, bvec_tensor])

    # Initial B-vector convolution
    init_tensor = DistributedConv3D(200, 1, name='init', instance_norm=True)(
        qspace_tensor
    )

    # Subsequent convolutional layers to extract features
    conv11_tensor = DistributedConv3D(376, 1, name='ecv11', instance_norm=True)(
        init_tensor
    )
    conv1_tensor = layers.Concatenate()([conv11_tensor, qspace_tensor])

    # Second convolutional block
    conv21_tensor = DistributedConv3D(664, 1, name='ecv21', batch_norm=True)(
        conv1_tensor
    )
    conv2_tensor = layers.Concatenate()([conv21_tensor, qspace_tensor])

    # Compress to latent space
    latent_tensor = DistributedConv3D(32, 1, name='ecvl1', batch_norm=True)(
        conv2_tensor
    )
    latent_tensor = DistributedConv3D(88, 1, name='ecvl2', batch_norm=True)(
        latent_tensor
    )

    out_tensor = layers.ConvLSTM3D(lstm_size, 1, name='conv_lstm')(latent_tensor)

    encoder = keras.models.Model(
        inputs=[input_img, input_vec], outputs=out_tensor, name='encoder'
    )

    return encoder


def get_1d_decoder(**kwargs):
    '''Genereates the decoder. This is 1D in the sense that all operations are pointwise.

    Keyword Args:
        q_out (int): Number of output q samples in model, set as `None` to
            allow varying number of timesteps. Default: `None`.
        in_vox_shape (Tuple[int,int,int]): voxel input shape
            Default: (10, 10, 10)
        lstm_size (int): Number of units within the 3DConvLSTM.
            Default: 48

    Returns:
        decoder (tf.keras.models.Model): Decoder model.
    '''
    q_out = kwargs.setdefault('q_out', None)
    in_vox_shape = kwargs.setdefault('in_vox_shape', (10, 10, 10))
    lstm_size = kwargs.setdefault('lstm_size', 48)
    latent_shape = in_vox_shape + (lstm_size,)

    # Inputs
    input_state = keras.Input(shape=(latent_shape), name='state')
    input_vecs = keras.Input(shape=(q_out, 3), name='bvec')

    # Decoder
    state_tensor = RepeatTensor()(input_state, 1, input_vecs, 1)
    bvec_tensor = RepeatBVector()(input_vecs, in_vox_shape)
    latent_tensor = layers.Concatenate()([state_tensor, bvec_tensor])

    # Initial B-vector convolution
    latent_tensor = DistributedConv3D(176, 1, name='dcvl1', batch_norm=True)(
        latent_tensor
    )
    latent_tensor = DistributedConv3D(224, 1, name='dcvl2', batch_norm=True)(
        latent_tensor
    )
    latent_tensor = layers.Concatenate()([latent_tensor, bvec_tensor])

    # Subsequent convolutional layers to extract features
    conv11_tensor = DistributedConv3D(632, 1, name='dcv11', batch_norm=True)(
        latent_tensor
    )
    conv1_tensor = layers.Concatenate()([conv11_tensor, bvec_tensor])

    # Second convolutional block
    conv21_tensor = DistributedConv3D(400, 1, name='dcv21', batch_norm=True)(
        conv1_tensor
    )
    conv2_tensor = layers.Concatenate()([conv21_tensor, bvec_tensor])

    conv3_tensor = DistributedConv3D(16, 1, name='dcv3')(conv2_tensor)

    out_tensor = DistributedConv3D(1, 1, name='dc4', activation='relu')(conv3_tensor)

    out_tensor = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(out_tensor)

    # Create decoder model instance
    decoder = keras.models.Model(
        inputs=[input_state, input_vecs], outputs=out_tensor, name='decoder'
    )

    return decoder


def get_1d_autoencoder(weights=None, **kwargs):
    '''Generates Autoencoder Model.
        Model is 1D in the sense that all operations are pointwise.

    Args:
        weights (str): Path to saved weights, leave as `None`
            to omit loading weights.

    Keyword Args:
        q_in (int): Number of input q samples in model, set as `None` to
            allow varying number of timesteps. Default: `None`.
        q_out (int): Number of output q samples in model, set as `None` to
            allow varying number of timesteps. Default: `None`.
        in_vox_shape (Tuple[int,int,int]): voxel input shape
            Default: (10, 10, 10)
        lstm_size (int): Number of units within the 3DConvLSTM.
            Default: 48
        loss (Union[str, tf.keras.losses.Loss]): Loss used in model training.
            see https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
            for example inputs. Default: "mae"

    Returns:
        autoencoder (tf.keras.models.Model): Autoencoder model.
    '''
    q_in = kwargs.setdefault('q_in', None)
    q_out = kwargs.setdefault('q_out', None)
    in_vox_shape = kwargs.setdefault('in_vox_shape', (10, 10, 10))
    kwargs.setdefault('lstm_size', 48)
    loss = kwargs.get('loss', 'mae')

    # Instantiate encoder & decoder models
    encoder = get_1d_encoder(**kwargs)
    decoder = get_1d_decoder(**kwargs)

    # Define inputs
    encoder_imgs = keras.Input(shape=(q_in,) + in_vox_shape, name='enc_images')
    encoder_vecs = keras.Input(shape=(q_in, 3), name='enc_bvecs')
    decoder_vecs = keras.Input(shape=(q_out, 3), name='dec_bvecs')

    # Connect encoder and decoder
    state = encoder([encoder_imgs, encoder_vecs])
    out_imgs = decoder([state, decoder_vecs])

    autoencoder = keras.models.Model(
        inputs=[encoder_imgs, encoder_vecs, decoder_vecs],
        outputs=out_imgs,
        name='autoencoder',
    )

    autoencoder.compile(loss=loss, optimizer=get_adam_opt())

    if weights is not None:
        autoencoder.load_weights(weights).assert_consumed()

    return autoencoder
