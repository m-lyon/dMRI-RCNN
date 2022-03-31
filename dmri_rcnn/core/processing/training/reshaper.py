'''tf.Graph class for reshaping arrays into input & output sets'''

import tensorflow as tf

from .base import DatasetMapper


class Reshaper(DatasetMapper):
    '''Splits input & output sets in intrashell manner using tf.Graph'''

    def __init__(self, in_num, out_num):
        '''Intrashell I/O splitter using dataset mapping

        Args:
            in_num (int): Number of qspace vols to use as input per
            out_num (int): Number of qspace vols to use as output

        Call Arguments:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                {
                    `shell`: {
                        'dmri': (tf.Tensor) shape -> (N, m, n, o, fs) dtype -> tf.float32,
                        'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32,
                    },
                }

        Output Spec:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                {
                    `shell`: {
                        'dmri_in': (tf.Tensor) shape -> (X, q_in, m, n, o) dtype -> tf.float32,
                        'dmri_out': (tf.Tensor) shape -> (X, q_out, m, n, o) dtype -> tf.float32,
                        'bvec_in': (tf.Tensor) shape -> (X, q_in, 3) dtype -> tf.float32,
                        'bvec_out': (tf.Tensor) shape -> (X, q_out, 3) dtype -> tf.float32,
                    },
                }
        '''
        self.q_in = tf.constant(in_num)
        self.q_out = tf.constant(out_num)
        self.subset_size = tf.constant(in_num + out_num)

    @tf.function
    def apply(self, data):
        '''Applies splitting algorithm

        Args:
            data (Dict[int,Dict]):
                `shell`: (Dict[str,tf.tensor])
                    'dmri': (tf.Tensor) shape -> (N, m, n, o, fs) dtype -> tf.float32
                    'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32

        Returns:
            data_out (Dict[int,Dict]):
                `shell`: (Dict[str,tf.tensor])
                    'dmri_in': (tf.Tensor) shape -> (X, q_in, m, n, o) dtype -> tf.float32
                    'dmri_out': (tf.Tensor) shape -> (X, q_out, m, n, o) dtype -> tf.float32
                    'bvec_in': (tf.Tensor) shape -> (X, q_in, 3) dtype -> tf.float32
                    'bvec_out': (tf.Tensor) shape -> (X, q_out, 3) dtype -> tf.float32
        '''

        data_out = {}
        for shell in data:
            data_out[shell] = {}

            dmri, bvec = data[shell]['dmri'], data[shell]['bvec']

            # Remove remainder qspace vols
            dmri, bvec = self._remove_remainders(dmri, bvec, self.subset_size)

            num_patches = tf.shape(dmri)[0]
            num_sets = tf.shape(bvec)[-1] // self.subset_size

            # Rearrange bvec
            bvec = self._rearrange_bvec(bvec, num_patches, num_sets, self.subset_size)
            bvec = tf.reshape(bvec, (-1, self.subset_size, 3))  # -> (X, fss, 3)

            # Rearrange dMRI
            dmri = self._rearrange_dmri(dmri, num_patches, num_sets)

            # Split into input & output sets
            data_out[shell]['dmri_in'] = dmri[:, 0 : self.q_in, ...]
            data_out[shell]['dmri_out'] = dmri[:, self.q_in :, ...]
            data_out[shell]['bvec_in'] = bvec[:, 0 : self.q_in, :]
            data_out[shell]['bvec_out'] = bvec[:, self.q_in :]

        return data_out

    @tf.function
    def _remove_remainders(self, dmri, bvec, subset_size):
        '''Removes remainder volumes if present

        Args:
            dmri (tf.Tensor): shape -> (N, m, n, o, fs), dtype -> tf.float32
            bvec (tf.Tensor): shape -> (3, fs), dtype -> tf.float32
            subset_size (tf.Tensor): shape -> (), dtype -> tf.int32

        Returns:
            dmri (tf.Tensor): shape -> (N, m, n, o, fs-c), dtype -> tf.float32
            bvec (tf.Tensor): shape -> (3, fs-c), dtype -> tf.float32
        '''
        rem = tf.shape(dmri)[-1] % subset_size

        if rem > 0:
            # Remove from end of arrays
            dmri = dmri[..., :-rem]
            bvec = bvec[:, :-rem]

        return dmri, bvec

    @tf.function
    def _rearrange_bvec(self, bvec, num_patches, num_sets, set_size):
        '''Rearranges bvec ready to be sliced to I/O sets

        Args:
            bvec (tf.Tensor): shape -> (3, fs-c), dtype -> tf.float32
            num_patches (tf.Tensor): shape -> (), dtype -> tf.int32
            num_sets (tf.Tensor): shape -> (), dtype -> tf.int32
            set_size (tf.Tensor): Size of set. shape -> (), dtype -> tf.int32

        Returns:
            bvec (tf.Tensor): shape -> (N, S, fss, 3), dtype -> tf.float32
        '''
        bvec = tf.expand_dims(bvec, axis=0)  # -> (1, 3, fs)
        bvec = tf.repeat(bvec, num_patches, axis=0)  # -> (N, 3, fs)
        bvec = tf.reshape(bvec, (num_patches, 3, num_sets, set_size))  # -> (N, 3, S, fss)
        bvec = tf.transpose(bvec, perm=[0, 2, 3, 1])  # -> (N, S, fss, 3)

        return bvec

    @tf.function
    def _rearrange_dmri(self, dmri, num_patches, num_sets):
        '''Rearranges bvec ready to be sliced to I/O sets

        Args:
            dmri (tf.Tensor): shape -> (N, m, n, o, fs-c) dtype -> tf.float32
            num_patches (tf.Tensor): shape -> (), dtype -> tf.int32
            num_sets (tf.Tensor): shape -> (), dtype -> tf.int32

        Returns:
            dmri (tf.Tensor): shape -> (X, fss, m, n, o), dtype -> tf.float32
        '''
        # pylint: disable=invalid-name
        m, n, o = tf.shape(dmri)[1], tf.shape(dmri)[2], tf.shape(dmri)[3]

        dmri = tf.reshape(dmri, (num_patches, m, n, o, num_sets, self.subset_size))
        dmri = tf.transpose(dmri, perm=[0, 4, 5, 1, 2, 3])
        dmri = tf.reshape(dmri, (-1, self.subset_size, m, n, o))

        return dmri


class Joiner(DatasetMapper):
    '''Joins I/O sets across shells using tf.Graph'''

    def __init__(self):
        '''Joins I/O sets across shells using tf.Graph

        Call Arguments:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                {
                    `shell`: {
                        'dmri_in': (tf.Tensor) shape -> (X, q_in, m, n, o) dtype -> tf.float32,
                        'dmri_out': (tf.Tensor) shape -> (X, q_out, m, n, o) dtype -> tf.float32,
                        'bvec_in': (tf.Tensor) shape -> (X, q_in, 3) dtype -> tf.float32,
                        'bvec_out': (tf.Tensor) shape -> (X, q_out, 3) dtype -> tf.float32,
                    },
                }

        Output Spec:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                {
                    'dmri_in': (tf.Tensor) shape -> (X*S, q_in, m, n, o) dtype -> tf.float32,
                    'dmri_out': (tf.Tensor) shape -> (X*S, q_out, m, n, o) dtype -> tf.float32,
                    'bvec_in': (tf.Tensor) shape -> (X*S, q_in, 3) dtype -> tf.float32,
                    'bvec_out': (tf.Tensor) shape -> (X*S, q_out, 3) dtype -> tf.float32,
                }
        '''

    @tf.function
    def apply(self, data):
        '''Applies joining algorithm

        Args:
            data (Dict[int,Dict]):
                `shell`: (Dict[str,tf.tensor])
                    'dmri_in': (tf.Tensor) shape -> (X, q_in, m, n, o) dtype -> tf.float32
                    'dmri_out': (tf.Tensor) shape -> (X, q_out, m, n, o) dtype -> tf.float32
                    'bvec_in': (tf.Tensor) shape -> (X, q_in, 3) dtype -> tf.float32
                    'bvec_out': (tf.Tensor) shape -> (X, q_out, 3) dtype -> tf.float32

        Returns:
            data_out (Dict[int,Dict]):
                'dmri_in': (tf.Tensor) shape -> (X*S, q_in, m, n, o) dtype -> tf.float32
                'dmri_out': (tf.Tensor) shape -> (X*S, q_out, m, n, o) dtype -> tf.float32
                'bvec_in': (tf.Tensor) shape -> (X*S, q_in, 3) dtype -> tf.float32
                'bvec_out': (tf.Tensor) shape -> (X*S, q_out, 3) dtype -> tf.float32
        '''
        data_out = {}
        # Combine across shells
        data_out['dmri_in'] = tf.concat([data[shell]['dmri_in'] for shell in data], 0)
        data_out['dmri_out'] = tf.concat([data[shell]['dmri_out'] for shell in data], 0)
        data_out['bvec_in'] = tf.concat([data[shell]['bvec_in'] for shell in data], 0)
        data_out['bvec_out'] = tf.concat([data[shell]['bvec_out'] for shell in data], 0)

        return data_out
