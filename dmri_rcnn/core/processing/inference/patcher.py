'''Patcher operation class'''

import numpy as np
import einops as ein

from .base import Operation


class Patcher(Operation):
    '''Splits data into 3D patches'''

    @staticmethod
    def _get_padding(orig_shape, patch_shape):
        '''Calculates padding needed to ensure whole 3D volume
            can be sliced into 3D patches of shape `patch_shape`.

        Args:
            orig_shape (Tuple[int,int,int]): (i, j, k)
            patch_shape (Tuple[int,int,int]): (m, n, o)

        Returns:
            padding (Tuple[Tuple[int,int]]): Nested tuple of padding
                length (0, pad) for each spatial dimension
        '''
        padding = ()
        for idx, size in enumerate(orig_shape):
            patch_size = patch_shape[idx]
            pad = size % patch_size
            if pad == 0:
                padding += ((0, 0),)
            else:
                total_pad = patch_size - pad
                if total_pad % 2 == 0:
                    padding += ((total_pad // 2, total_pad // 2),)
                else:
                    padding += ((total_pad // 2, (total_pad // 2) + 1),)

        return padding

    @staticmethod
    def _apply_padding(data_array, padding):
        '''Applies padding to data array.

        Args:
            data_array (np.ndarray): array containing data
                with data_array.ndim >= 3 and first three
                dimensions correspond to (i, j, k)
            padding (Tuple[Tuple[int,int]]): Nested tuple of padding
                length (inner, outer) for each spatial dimension

        Returns:
            data_array (np.ndarray): modified array
                with dimensions of (i+di, j+dj, k+dk, ...)
        '''
        # Get extra dims padding
        extra_dim_pads = tuple([(0, 0) for _ in data_array.shape[3:]])

        # Apply padding
        data_array = np.pad(data_array, padding + extra_dim_pads)

        return data_array

    @staticmethod
    def _remove_padding(data_array, padding):
        '''Removes padding from data_array

        Args:
            data_array (np.ndarray): data array
                with dimensions of (i+di, j+dj, k+dk, ...)
            padding (Tuple[Tuple[int,int]]): Nested tuple of padding
                length (inner, outer) for each spatial dimension

        Returns:
            data_array (np.ndarray): modified array
                shape -> (i, j, k, ...)
        '''
        data_array = data_array[
            padding[0][0] or None : -padding[0][1] or None,
            padding[1][0] or None : -padding[1][1] or None,
            padding[2][0] or None : -padding[2][1] or None,
        ]

        return data_array

    @staticmethod
    def _combine_data(data_array, padding, orig_shape):
        '''Combine data from patches into contiguous 3D volumes

        Args:
            data_array (np.ndarray): shape -> (X, m, n, o, fs)
            padding (Tuple[Tuple[int,int]]): Nested tuple of padding
                length (inner, outer) for each spatial dimension
            orig_shape (Tuple[int,int,int]): Original spatial
                dimensions (i, j, k)

        Returns:
            data_array (np.ndarray): shape -> (i+padi, j+padj, k+padk, fs)
        '''
        nums = ()
        for idx, pad in enumerate(padding):
            orig_size = orig_shape[idx]
            patch_size = data_array.shape[idx + 1]

            new_size = orig_size + pad[0] + pad[1]
            assert new_size % patch_size == 0

            nums += (new_size // patch_size,)

        data_array = ein.rearrange(
            data_array,
            '(M N O) m n o fs -> (M m) (N n) (O o) fs',
            M=nums[0],
            N=nums[1],
            O=nums[2],
        )

        return data_array

    @classmethod
    def forward(cls, datasets, context, **kwargs):
        '''Slices data into patches of size determined by `patch_shape_in`

        Args:
            datasets (Dict[str,Any]):
                'mask': (np.ndarray) -> shape (i, j, k)
                'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                ...
            context (Dict[str,Any]):
                ...

        Keyword Args:
            patch_shape_in: (Tuple[int,int,int])

        Modifies:
            datasets (Dict[str,Any]):
                - 'mask': (np.ndarray) -> shape (i, j, k)
                ~ 'dmri_in': (np.ndarray) -> shape (X, m, n, o, q_in)
                ...

            context (Dict[str,Any]):
                + 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                + 'padding': (Tuple[Tuple[int,int,int]]) -> (0, padi), (0, padj), (0, padk)
                + 'mask_filter': (np.ndarray) -> shape (N,)
                + 'unused_num': (int)
                ...
        '''
        # pylint: disable=invalid-name
        print('Slicing data into 3D patches...')
        mask = datasets.pop('mask')
        context['orig_shape'] = mask.shape
        m, n, o = kwargs['patch_shape_in']

        # Get padding
        context['padding'] = cls._get_padding(mask.shape, kwargs['patch_shape_in'])

        # Pad mask
        mask = cls._apply_padding(mask, context['padding'])

        # Rearrange mask
        mask = ein.rearrange(
            mask, '(mx m) (nx n) (ox o) -> (mx nx ox) m n o', m=m, n=n, o=o
        )

        # Filter out patches that are not contained within brain mask
        mask_filter = np.sum(mask, (1, 2, 3), dtype=bool)
        context['mask_filter'] = mask_filter

        # Pad dMRI
        datasets['dmri_in'] = cls._apply_padding(
            datasets['dmri_in'], context['padding']
        )

        # Slice into patches
        datasets['dmri_in'] = ein.rearrange(
            datasets['dmri_in'],
            '(mx m) (nx n) (ox o) q_in -> (mx nx ox) m n o q_in',
            n=n,
            m=m,
            o=o,
        )

        # Save unused patches
        unused_patches = datasets['dmri_in'][~mask_filter, ...]
        context['unused_num'] = unused_patches.shape[0]

        # Apply mask filter
        datasets['dmri_in'] = datasets['dmri_in'][mask_filter, ...]

    @classmethod
    def backward(cls, datasets, context, **kwargs):
        '''Combines 3D patches into 3D whole volumes

        Args:
            datasets (Dict[str,Any]):
                'dmri_out': (np.ndarray) -> shape (X, m, n, o, q_out)
            context (Dict[str,Any]):
                'padding': (Tuple[Tuple[int,int,int]]) -> (0, padi), (0, padj), (0, padk)
                'mask_filter': (np.ndarray) -> shape (N,)
                'orig_shape': (Tuple[int,int,int]) -> i, j, k
                'unused_num': (int)
                ...

        Modifies:
            datasets (Dict[str,Any]):
                ~ 'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
                ...
            context (Dict[str,Any]):
                - 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                - 'padding': (Tuple[Tuple[int,int,int]]) -> (0, padi), (0, padj), (0, padk)
                - 'mask_filter': (np.ndarray) -> shape (N,)
                - 'unused_num': (int)
                ...
        '''
        # pylint: disable=invalid-name
        print('Combining 3D patches into contiguous volumes...')

        mask_filter = context.pop('mask_filter')
        orig_shape = context.pop('orig_shape')
        padding = context.pop('padding')
        unused_num = context.pop('unused_num')

        idx_pos = np.arange(len(mask_filter))[mask_filter]
        idx_neg = np.arange(len(mask_filter))[~mask_filter]
        order = np.argsort(np.concatenate([idx_neg, idx_pos]))

        if unused_num:
            _, m, n, o, q_out = datasets['dmri_out'].shape
            unused_patches = np.zeros((unused_num, m, n, o, q_out), dtype=np.float32)

            datasets['dmri_out'] = np.concatenate(
                [unused_patches, datasets['dmri_out']], axis=0
            )

        # Re-order patches
        datasets['dmri_out'] = datasets['dmri_out'][order, ...]

        # Recombine
        datasets['dmri_out'] = cls._combine_data(
            datasets['dmri_out'], padding, orig_shape
        )

        # Remove padding
        datasets['dmri_out'] = cls._remove_padding(datasets['dmri_out'], padding)
