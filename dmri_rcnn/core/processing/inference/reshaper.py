'''Input & Output reshaper class'''

import numpy as np

from .base import Operation


class Reshaper(Operation):
    '''Reshapes dMRI data and modifies axis order'''

    @staticmethod
    def forward(datasets):
        '''Reshapes and repeats data

        Args:
            datasets (Dict[str,Any]):
                'dmri_in': (np.ndarray) -> shape (X, m, n, o, q_in)
                'bvec_in': (np.ndarray) -> shape (3, q_in)
                'bvec_out': (np.ndarray) -> shape (3, q_out)
                ...

        Modifies:
            datasets (Dict[str,Any]):
                ~ 'dmri_in': (np.ndarray) -> shape (X, q_in, m, n, o)
                ~ 'bvec_in': (np.ndarray) -> shape (X, q_in, 3)
                ~ 'bvec_out': (np.ndarray) -> shape (X, q_out, 3)
                ...
        '''
        # pylint: disable=arguments-differ
        # Reshape dmri_in -> (X, q_in, m, n, o)
        datasets['dmri_in'] = np.transpose(datasets['dmri_in'], axes=[0, 4, 1, 2, 3])
        x_len = datasets['dmri_in'].shape[0]

        # Reshape & repeat bvec_in
        bvec_in = datasets['bvec_in'].T
        datasets['bvec_in'] = np.tile(bvec_in[np.newaxis, :, :], reps=(x_len, 1, 1))

        # Reshape & repeat bvec_out
        bvec_out = datasets['bvec_out'].T
        datasets['bvec_out'] = np.tile(bvec_out[np.newaxis, :, :], reps=(x_len, 1, 1))

    @staticmethod
    def backward(datasets):
        '''Reshapes dMRI

        Args:
            datasets (Dict[str,Any]):
                'dmri_out': (np.ndarray) -> shape (X, q_out, m, n, o)

        Modifies:
            datasets (Dict[str,Any]):
                ~ 'dmri_out': (np.ndarray) -> shape (X, m, n, o, q_out)
        '''
        # pylint: disable=arguments-differ
        # Reshape dmri_out -> (X, m, n, o, q_out)
        datasets['dmri_out'] = np.transpose(datasets['dmri_out'], axes=[0, 2, 3, 4, 1])
