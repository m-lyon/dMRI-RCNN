'''tf.Graph Patcher operation class'''

import tensorflow as tf

from .base import DatasetMapper


class Patcher(DatasetMapper):
    '''3D Patcher using tf.Graph implementation'''

    def __init__(self, patch_shape):
        '''3D patcher dataset mapping

        Args:
            patch_shape (Tuple[int,int,int]): 3D patch size.

        Call Arguments:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                {
                    'mask': (tf.Tensor) shape -> (i, j, k),
                    'data_use': {
                        `shell`: {
                            'dmri': (tf.Tensor) shape -> (i, j, k, fs) dtype -> tf.float32,
                            'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32,
                        }
                    },
                }

        Output Spec:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                {
                    `shell`: {
                        'dmri': (tf.Tensor) shape -> (N, m, n, o, fs) dtype -> tf.float32,
                        'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32,
                    }
                }
        '''
        self.patch_shape = tf.constant(patch_shape)

    @tf.function
    def apply(self, data):
        '''Applies algorithm to data

        Args:
            data (Dict[Any,Any]):
                'mask': (tf.Tensor) shape -> (i, j, k) dtype -> tf.int8
                'data_use': (Dict[int,Dict])
                    `shell`: (Dict[str,tf.tensor])
                        'dmri': (tf.Tensor) shape -> (i, j, k, fs) dtype -> tf.float32
                        'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32

        Returns:
            data_out (Dict[int,Dict]):
                `shell`: (Dict[str,tf.tensor])
                    'dmri': (tf.Tensor) shape -> (N, m, n, o, fs) dtype -> tf.float32
                    'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32
        '''
        data_out = {}
        mask = data['mask']

        # Get padding
        padding = self._get_padding(mask)

        # Apply padding to mask
        mask = self._apply_padding(mask, padding)

        # Get mask filter
        mask_filter = self._get_mask_filter(mask)

        for shell in data['data_use']:
            data_out[shell] = {}

            dmri = data['data_use'][shell]['dmri']

            # Apply padding to dMRI
            dmri = self._apply_padding(dmri, padding)

            # Reshape dMRI
            dmri = self._reshape_dmri(dmri)

            # Apply filter to dMRI
            dmri = tf.boolean_mask(dmri, mask_filter, axis=0)

            data_out[shell]['dmri'] = dmri
            data_out[shell]['bvec'] = data['data_use'][shell]['bvec']

        return data_out

    @tf.function
    def _get_padding(self, tensor):
        '''Calculates padding needed to ensure whole 3D volume
            can be sliced into 3D patches of shape `patch_shape`.

        Args:
            tensor (tf.Tensor): shape -> (i, j, k)

        Returns:
            padding (tf.Tensor): shape -> (3, 2), dtype -> tf.int32
        '''
        pad = tf.TensorArray(dtype=tf.int32, size=tf.rank(tensor))
        tensor_shape = tf.shape(tensor)

        for idx in tf.range(tf.rank(tensor)):
            rem = tensor_shape[idx] % self.patch_shape[idx]

            if tf.equal(0, rem):
                pad = pad.write(idx, tf.constant([0, 0]))
            else:
                total_pad = self.patch_shape[idx] - rem
                if total_pad % 2 == 0:
                    pad = pad.write(idx, tf.stack([total_pad // 2, total_pad // 2]))
                else:
                    pad = pad.write(
                        idx, tf.stack([total_pad // 2, (total_pad // 2) + 1])
                    )

        return pad.stack()

    @tf.function
    def _apply_padding(self, tensor, padding):
        '''Applies padding to tensor first 3 dimensions.

        Args:
            tensor (tf.Tensor): tensor containing data
                with tensor.ndim >= 3 and first three
                dimensions correspond to (i, j, k)
            padding (tf.Tensor): shape -> (3, 2), dtype -> tf.int32

        Returns:
            tensor (np.ndarray): modified tensor
                with dimensions of (i+di, j+dj, k+dk, ...)
        '''
        # Get extra dims padding
        extra_dims = tf.rank(tensor) - 3
        extra_pad = tf.TensorArray(dtype=tf.int32, size=extra_dims)

        for idx in tf.range(extra_dims):
            extra_pad = extra_pad.write(idx, tf.constant([0, 0]))

        padding = tf.concat(
            [padding, tf.reshape(extra_pad.stack(), (extra_dims, 2))], 0
        )

        # Apply padding
        tensor = tf.pad(tensor, padding, constant_values=0)

        return tensor

    @tf.function
    def _get_mask_filter(self, mask):
        '''Gets mask filter to remove empty patches

        Args:
            mask (tf.Tensor): shape -> (i, j, k), dtype -> tf.int8

        Returns:
            mask_filter (tf.Tensor): shape -> (N,), dtype -> tf.bool
        '''
        # Reshape mask
        mask = self._reshape_mask(mask)

        # Cast mask to int64
        mask = tf.cast(mask, tf.int64)

        # Filter out patches that are not contained within brain mask
        mask_filter = tf.reduce_sum(mask, axis=(1, 2, 3))
        mask_filter = tf.cast(mask_filter, tf.bool)

        return mask_filter

    @tf.function
    def _reshape_mask(self, mask):
        '''Reshapes mask into N patches of `self.patch_shape` shape

        Args:
            mask (tf.Tensor): shape -> (i, j, k), dtype -> tf.int8

        Returns:
            mask (tf.Tensor): shape -> (N, m, n, o), dtype -> tf.int8
        '''
        pnums = tf.shape(mask) // self.patch_shape
        pshape = self.patch_shape
        mask = tf.reshape(
            mask, (pnums[0], pshape[0], pnums[1], pshape[1], pnums[2], pshape[2])
        )
        mask = tf.transpose(mask, perm=[0, 2, 4, 1, 3, 5])
        mask = tf.reshape(mask, tf.concat([tf.constant([-1]), pshape], 0))

        return mask

    @tf.function
    def _reshape_dmri(self, dmri):
        '''Reshapes dMRI into N patches of `self.patch_shape` shape

        Args:
            dmri (tf.Tensor): shape -> (i, j, k, fs), dtype -> tf.float32

        Returns:
            dmri (tf.Tensor): shape -> (N, m, n, o, fs), dtype -> tf.float32
        '''
        pshape = tf.concat([self.patch_shape, [tf.shape(dmri)[-1]]], 0)
        pnums = tf.shape(dmri) // pshape
        dmri = tf.reshape(
            dmri, (pnums[0], pshape[0], pnums[1], pshape[1], pnums[2], pshape[2], -1)
        )
        dmri = tf.transpose(dmri, perm=[0, 2, 4, 1, 3, 5, 6])
        dmri = tf.reshape(dmri, tf.concat([tf.constant([-1]), pshape], 0))

        return dmri
