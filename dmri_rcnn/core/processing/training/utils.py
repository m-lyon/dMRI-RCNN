'''Utility functions for tf.Graph modules'''

import tensorflow as tf


@tf.function
def spherical_distances(x, y):
    '''Calculates spherical distance matrix between x and y

        See dipy.core.geometry.sphere_distance for implementation
            with varying radii.
    Args:
        x (tf.Tensor): array one -> shape (3, m)
        y (tf.Tensor): array two -> shape (3, n)

    Returns:
        sph_dist (tf.Tensor): spherical distance matrix
            shape -> (m, n)
    '''
    # pylint: disable=invalid-name

    # Normalise to unit vector
    x = x / tf.norm(x, axis=0)
    y = y / tf.norm(y, axis=0)

    # Dot product
    dots = tf.matmul(x, y, transpose_a=True)
    # Ensure no values above 1
    dots = tf.clip_by_value(dots, -1.0, 1.0)

    # Calculate distance
    dist = tf.acos(dots)

    return dist


@tf.function
def update_1d_array(array, vals, indices):
    '''Updates a 1D Array with `vals` at `indices`

    Args:
        array (tf.Tensor): 1D Tensor.
        vals (tf.Tensor or List): Values to update `array` with,
            entries in `vals` must be same type as in `array`.
        indices (tf.Tensor or List): Indices in `array` to update,
            dtype -> tf.int32

    Returns:
        (tf.Tensor): `array` with updates values.
    '''
    indices = tf.expand_dims(indices, axis=1)
    return tf.tensor_scatter_nd_update(array, indices, vals)
