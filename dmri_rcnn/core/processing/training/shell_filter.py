'''Shell Filter class'''

import tensorflow as tf

from .base import DatasetMapper


class ShellFilter(DatasetMapper):
    '''Filters shells using tf.Graph implementation'''

    def __init__(self, shells, shell_var=30.0):
        '''Shell filter dataset mapping

        Args:
            shells (List[int,]): List of shells to be kept.
            shell_var (float): Variance to threshold shell membership
                Default: 30.0

        Call Arguments:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                {
                    'dmri': (tf.Tensor) shape -> (i, j, k, b) dtype -> tf.float32,
                    'mask': (tf.Tensor) shape -> (i, j, k) dtype -> tf.int8,
                    'bvec': (tf.Tensor) shape -> (3, b) dtype -> tf.float32,
                    'bval': (tf.Tensor) shape -> (b,) dtype -> tf.float32,
                }

        Output Spec:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                {
                    'mask': (tf.Tensor) shape -> (i, j, k),
                    'data_use': {
                        `shell`: {
                            'dmri': (tf.Tensor) shape -> (i, j, k, fs) dtype -> tf.float32,
                            'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32,
                            'bval': (tf.Tensor) shape -> (fs,) dtype -> tf.float32,
                        }
                    },
                }
        '''
        self.shells = shells
        self.shell_var = tf.constant(shell_var, dtype=tf.float32)

    @tf.function
    def _get_shell_filter(self, shell, bval):
        '''Applies shell filtering with given shell variance

        Args:
            shell (tf.Tensor): Shell to select for, shape -> (), dtype -> tf.float3232
            bval (tf.Tensor): bval tensor, shape -> (b,) dtype -> tf.float32

        Returns:
            (tf.Tensor): shell filter mask, shape -> (b,), type -> tf.bool
        '''
        min_thresh = tf.greater_equal(bval, shell - self.shell_var)
        max_thresh = tf.less_equal(bval, shell + self.shell_var)

        return tf.logical_and(min_thresh, max_thresh)

    @tf.function
    def apply(self, data):
        '''Applies shell filtering algorithm

        Args:
            data (Dict[str,tf.tensor]):
                'dmri': (tf.Tensor) shape -> (i, j, k, b) dtype -> tf.float32
                'mask': (tf.Tensor) shape -> (i, j, k) dtype -> tf.int8
                'bvec': (tf.Tensor) shape -> (3, b) dtype -> tf.float32
                'bval': (tf.Tensor) shape -> (b,) dtype -> tf.float32

        Returns:
            data_out (Dict[Any,Any]):
                'mask': (tf.Tensor) shape -> (i, j, k) dtype -> tf.int8
                'data_use': (Dict[int,Dict])
                    `shell`: (Dict[str,tf.tensor])
                        'dmri': (tf.Tensor) shape -> (i, j, k, fs) dtype -> tf.float32
                        'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32
                        'bval': (tf.Tensor) shape -> (fs,) dtype -> tf.float32
        '''
        dmri, bvec, bval, mask = data['dmri'], data['bvec'], data['bval'], data['mask']

        data_out, data_use = {'mask': mask}, {}

        for shell in self.shells:
            data_use[shell] = {}
            shell_index = tf.range(tf.shape(bval)[0])

            shell_filter = self._get_shell_filter(
                tf.constant(shell, dtype=tf.float32), bval
            )
            take_filter = shell_index[shell_filter]

            data_use[shell]['dmri'] = tf.gather(dmri, take_filter, axis=-1)
            data_use[shell]['bvec'] = tf.gather(bvec, take_filter, axis=-1)
            data_use[shell]['bval'] = tf.gather(bval, take_filter, axis=-1)

        data_out['data_use'] = data_use

        return data_out
