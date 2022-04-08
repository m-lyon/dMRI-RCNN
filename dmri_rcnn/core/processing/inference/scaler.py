'''Scaler operation'''
import numpy as np

from .base import Operation


class Scaler(Operation):
    '''Scaler Operation'''

    @staticmethod
    def apply(data, xmin: float, xmax: float):
        '''Applies scaling in forward direction'''
        return (data - xmin) / (xmax - xmin)

    @staticmethod
    def reverse(data, xmin: float, xmax: float):
        '''Applies scaling in reverse direction'''
        return ((xmax - xmin) * data) + xmin

    @classmethod
    def forward(cls, datasets, context, **kwargs):
        '''Rescales dMRI data to range [0,1] independently in each shell.

        Args:
            datasets (Dict[str,Any]):
                'mask': (np.ndarray) -> shape (i, j, k)
                'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                'bvec_in': (np.ndarray) -> shape (3, q_in)
                'bvec_out': (np.ndarray) -> shape (3, q_out)
            context (Dict[str,Any]):
                ...

        Keyword Args:
            shell (int): Shell being processed
            norms (Dict[int,Tuple[float,float]]): Normalisations for each shell
            apply_bvec (bool): Apply normalisation to b-vector. Default: False

        Modifies:
            datasets (Dict[str,Any]):
                ~ 'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                ~ 'bvec_in': (np.ndarray) -> shape (3, q_in)
                ~ 'bvec_in': (np.ndarray) -> shape (3, q_out)
                ...
            context (Dict[str,Any]):
                + 'xmin': (float)
                + 'xmax': (float)
                ...
        '''
        print('Normalizing dMRI data...')

        # Get xmin xmax
        xmin, xmax = 0.0, kwargs['norms'][kwargs['shell']]

        # Apply rescaling
        datasets['dmri_in'] = cls.apply(datasets['dmri_in'], xmin, xmax)

        if kwargs.get('apply_bvec', False):
            datasets['bvec_in'] = datasets['bvec_in'] * kwargs['shell'] / 1000.0
            datasets['bvec_out'] = datasets['bvec_out'] * kwargs['shell'] / 1000.0

        # Save scale units
        context['xmin'] = xmin
        context['xmax'] = xmax

    @classmethod
    def backward(cls, datasets, context, **kwargs):
        '''Rescales dMRI data back to original intensity range

        Args:
            datasets (Dict[str,Any]):
                'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
            context (Dict[str,Any]):
                'xmin': (float)
                'xmax': (float)
                ...

        Modifies:
            datasets (Dict[str,Any]):
                ~ 'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
            context (Dict[str,Any]):
                - 'xmin': (float)
                - 'xmax': (float)
                ...
        '''
        print('Rescaling dMRI data to original intensity range...')
        xmin, xmax = context.pop('xmin'), context.pop('xmax')

        datasets['dmri_out'] = cls.reverse(datasets['dmri_out'], xmin, xmax)


class ScalerNorm(Scaler):
    '''Scaling with individual normalisation'''

    @classmethod
    def forward(cls, datasets, context, **kwargs):
        '''Rescales dMRI data to range [0,1] independently in each shell.

        Args:
            datasets (Dict[str,Any]):
                'mask': (np.ndarray) -> shape (i, j, k)
                'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                'bvec_in': (np.ndarray) -> shape (3, q_in)
                'bvec_out': (np.ndarray) -> shape (3, q_out)
            context (Dict[str,Any]):
                ...

        Keyword Args:
            shell (int): Shell being processed
            pcent (int): Maximum intensity percentile, as an int. e.g.
                99% = 99
            apply_bvec (bool): Apply normalisation to b-vector. Default: False

        Modifies:
            datasets (Dict[str,Any]):
                ~ 'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                ~ 'bvec_in': (np.ndarray) -> shape (3, q_in)
                ~ 'bvec_in': (np.ndarray) -> shape (3, q_out)
                ...
            context (Dict[str,Any]):
                + 'xmin': (float)
                + 'xmax': (float)
                ...
        '''
        print('Normalizing dMRI data...')
        pcent = kwargs['pcent']

        # Get xmin xmax
        xmin = 0.0
        xmax = np.percentile(datasets['dmri_in'][datasets['mask'].astype(np.bool)], pcent)

        # Apply rescaling
        datasets['dmri_in'] = cls.apply(datasets['dmri_in'], xmin, xmax)

        if kwargs.get('apply_bvec', False):
            datasets['bvec_in'] = datasets['bvec_in'] * kwargs['shell'] / 1000.0
            datasets['bvec_out'] = datasets['bvec_out'] * kwargs['shell'] / 1000.0

        # Save scale units
        context['xmin'] = xmin
        context['xmax'] = xmax
