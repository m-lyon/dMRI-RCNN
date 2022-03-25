'''tf.Graph Scaling data class'''

import tensorflow as tf

from .base import DatasetMapper


class DataScaler(DatasetMapper):
    '''DataScaler'''

    def __init__(self, norms):
        '''Data scaling dataset mapping

        Arguments:
            norms (Dict[int,Tuple[float,float]]): Normalisations for each shell
                {`shell`: (xmin, xmax)}

        Call Arguments:
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

        Output Spec:
            a tf.data.Dataset object with the same structure as input, with rescaled
                dmri and bvec values.
        '''
        self.norms = {
            key: (tf.constant(val[0], dtype=tf.float32), tf.constant(val[1], dtype=tf.float32))
            for (key, val) in norms.items()
        }

    @tf.function
    def apply(self, data):
        '''Applies algorithm to data

        Args:
            data_out (Dict[Any,Any]):
                'mask': (tf.Tensor) shape -> (i, j, k) dtype -> tf.int8
                'data_use': (Dict[int,Dict])
                    `shell`: (Dict[str,tf.tensor])
                        'dmri': (tf.Tensor) shape -> (i, j, k, fs) dtype -> tf.float32
                        'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32

        Returns:
            out_data (Dict[Any,Any]): Same entries as input, with rescaled dmri.
        '''
        out_data, data_use = {'mask': data['mask']}, {}
        for shell in data['data_use']:
            data_use[shell] = {}
            dmri = data['data_use'][shell]['dmri']

            xmin, xmax = self.norms[shell][0], self.norms[shell][1]
            dmri = self._apply_rescale_dmri(dmri, xmin, xmax)

            if shell == 0:
                dmri = tf.reduce_mean(dmri, axis=-1, keepdims=True)

            data_use[shell]['dmri'] = dmri
            data_use[shell]['bvec'] = data['data_use'][shell]['bvec']

        out_data['data_use'] = data_use

        return out_data

    @tf.function
    def _apply_rescale_dmri(self, dmri, xmin, xmax):
        '''Apples rescaling to dMRI using given `xmin` and `xmax`

        Args:
            dmri (tf.Tensor): shape -> (i, j, k, fs) dtype -> tf.float32
            xmin (tf.Tensor): shape -> () dtype -> tf.float32
            xmin (tf.Tensor): shape -> () dtype -> tf.float32

        Returns:
            (tf.Tensor): dMRI rescaled, shape -> (i, j, k, fs) dtype -> tf.float32
        '''
        return tf.divide(tf.subtract(dmri, xmin), tf.subtract(xmax, xmin))
