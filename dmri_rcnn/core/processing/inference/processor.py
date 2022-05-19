'''Inference processor class'''
import numpy as np
import tensorflow as tf

from ...io import load_nifti, load_bvec, save_nifti

from .scaler import Scaler, ScalerNorm
from .patcher import Patcher
from .reshaper import Reshaper


class InferenceProcessor:
    '''Inference pipeline for single-shell data'''

    def __init__(self, model, **kwargs):
        '''Initialises processor object

        Args:
            model (tf.keras.models.Model): Compiled model with weights
                already loaded.
        Keyword Args:
            shell (int): Selected shell
                Default: 1000
            patch_shape_in (Tuple[int,int,int]): shape of input patches to train/infer with
                Default: determined by `model`.
            norms (Dict[int,Tuple[float,float]]): Normalisations for each shell
                Default: {1000: 4000.0, 2000: 3000.0, 3000: 2000.0}
            batch_size (int): Batch size of dataset for inference/training.
                Default: 4
        '''
        self.model = model
        self._scaler = Scaler
        self._patcher = Patcher
        self._reshaper = Reshaper
        self._config = self._get_config(kwargs)

    def _get_config(self, kwargs):
        '''Sets internal default keyword args `self._config`'''
        config = {
            'shell': kwargs.get('shell', 1000),
            'patch_shape_in': kwargs.get(
                'patch_shape_in', self.model.input[0].shape[-3:]
            ),
            'norms': kwargs.get('norms', {1000: 4000.0, 2000: 3000.0, 3000: 2000.0}),
            'batch_size': kwargs.get('batch_size', 4),
        }
        return config

    def assert_dmri_input(self, dmri_in):
        '''Asserts whether dMRI input is the right shape for the model'''
        return self.model.input[0].shape.assert_is_compatible_with(dmri_in.shape)

    def assert_vec_input(self, vec_in):
        '''Asserts whether B-vector input is the right shape for the model'''
        return self.model.input[1].shape.assert_is_compatible_with(vec_in.shape)

    def assert_vec_output(self, vec_out):
        '''Asserts whether B-vector input is the right shape for the model'''
        return self.model.input[2].shape.assert_is_compatible_with(vec_out.shape)

    def print_config(self):
        '''Prints configuration of processor'''
        print('Configuration:')
        for key, val in self._config.items():
            print(f'    {key} -> {val}')

    @staticmethod
    def load_raw_data_dict(dmri_in, bvec_in, bvec_out, mask):
        '''Loads dMRI data into memory and zips to datasets dict

        Args:
            dmri_in (str): Path to input dMRI data file.
            bvec_in (str): Path to input b-vector data file.
            bvec_out (str): Path to output b-vector data file.
            mask (str): Path to brain mask file.

        Returns:
            datasets (Dict[str,Any]):
                'mask': (np.ndarray) -> shape (i, j, k)
                'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                'bvec_in': (np.ndarray) -> shape (3, q_in)
                'bvec_out': (np.ndarray) -> shape (3, q_out)
            context (Dict[str,Any]):
                'affine': (np.ndarray) -> shape (4, 4)
        '''
        dmri_in, affine = load_nifti(dmri_in)
        mask, _ = load_nifti(mask, dtype=np.int8)
        bvec_in, bvec_out = load_bvec(bvec_in), load_bvec(bvec_out)

        datasets = {
            'mask': mask,
            'dmri_in': dmri_in,
            'bvec_in': bvec_in,
            'bvec_out': bvec_out,
        }
        context = {'affine': affine}

        return datasets, context

    def run_subject(self, dmri_in, bvec_in, bvec_out, mask, dmri_out=None):
        '''Runs subject through preprocessing, single-shell inference,
            and postprocessing.

        Args:
            dmri_in (str): Path to input dMRI data file.
            bvec_in (str): Path to input b-vector data file.
            bvec_out (str): Path to output b-vector data file.
            mask (str): Path to brain mask file.
            dmri_out (str): Optional output path to save dmri_out to disk.
                Default: `None` (does not save to disk)

        Returns:
            dmri_out (np.ndarray): Inferred dMRI data
        '''
        datasets, context = self.load_raw_data_dict(dmri_in, bvec_in, bvec_out, mask)
        self.preprocess(datasets, context)
        self.run_model(datasets)
        self.postprocess(datasets, context)
        if dmri_out is not None:
            self.save_dmri_data(datasets, context, dmri_out)
        print('Done.')
        return datasets['dmri_out']

    def preprocess(self, datasets, context):
        '''Preprocessing for a given subject:
            1) Rescales data to approximately [0,1] range
            2) Slices data into smaller 3D patches
            3) Reshapes data ready for inference

        Args:
            datasets (Dict[str,Any]):
                'mask': (np.ndarray) -> shape (i, j, k)
                'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                'bvec_in': (np.ndarray) -> shape (3, q_in)
                'bvec_out': (np.ndarray) -> shape (3, q_out)
            context (Dict[str,Any]):
                'affine': (np.ndarray) -> shape (4, 4)

        Modifies:
            datasets (Dict[str,Any]):
                ~ 'dmri_in': (np.ndarray) -> shape (X, q_in, m, n, o)
                ~ 'bvec_in': (np.ndarray) -> shape (X, q_in, 3)
                ~ 'bvec_out': (np.ndarray) -> shape (X, q_out, 3)
                - 'mask': (np.ndarray) -> shape (i, j, k)
            context (Dict[str,Any])
                + 'xmin': (float)
                + 'xmax': (float)
                + 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                + 'padding': (Tuple[Tuple[int,int,int]]) -> (0, padi), (0, padj), (0, padk)
                + 'mask_filter': (np.ndarray) -> shape (N,)
                + 'unused_num': (int)
                ...
        '''
        # Scales data to approximate range [0,1]
        self._scaler.forward(datasets, context, **self._config)
        # Splits dMRI data into 3D patches
        self._patcher.forward(datasets, context, **self._config)
        # Reshapes and repeats dataset
        self._reshaper.forward(datasets)

    def postprocess(self, datasets, context):
        '''Postprocessing for a given subject:
            1) Reshapes data back to previous shapes
            2) Combines patches into contiguous 4D volumes
            1) Rescales data back to original value range.

        Args:
            datasets (Dict[str,Any]):
                'dmri_out': (np.ndarray) -> shape (X, q_out, n, m, o)
            context (Dict[str,Any]):
                'xmin': (float)
                'xmax': (float)
                'orig_shape': (Tuple[int,int,int]) -> i, j, k
                'padding': (Tuple[Tuple[int,int,int]]) -> (0, padi), (0, padj), (0, padk)
                'mask_filter': (np.ndarray) -> shape (N,)
                'unused_num': (int)
                ...

        Modifies:
            datasets (Dict[str,Any])
                ~ 'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
            context (Dict[str,Any]):
                - 'xmin': (float)
                - 'xmax': (float)
                - 'orig_shape': (Tuple[int,int,int]) -> i, j, k
                - 'padding': (Tuple[Tuple[int,int,int]]) -> (0, padi), (0, padj), (0, padk)
                - 'mask_filter': (np.ndarray) -> shape (N,)
                - 'unused_num': (int)
        '''
        # Reshape dMRI
        self._reshaper.backward(datasets)
        # Combines 3D patches to contiguous volume
        self._patcher.backward(datasets, context)
        # Rescale data
        self._scaler.backward(datasets, context)

    def run_model(self, datasets):
        '''Runs model through inference to produce dMRI outputs

        Args:
            datasets (Dict[str,Any]):
                'dmri_in': (np.ndarray) -> shape (X, q_in, m, n, o)
                'bvec_in': (np.ndarray) -> shape (X, q_in, 3)
                'bvec_out': (np.ndarray) -> shape (X, q_out, 3)

        Modifies:
            datasets (Dict[str,Any]):
                + 'dmri_out': (np.ndarray) -> shape (X, q_out, n, m, o)
                - 'dmri_in': (np.ndarray) -> shape (X, q_in, m, n, o)
                - 'bvec_in': (np.ndarray) -> shape (X, q_in, 3)
                - 'bvec_out': (np.ndarray) -> shape (X, q_out, 3)
        '''
        print('Running inference on data...')

        dmri_in = datasets.pop('dmri_in')
        bvec_in = datasets.pop('bvec_in')
        bvec_out = datasets.pop('bvec_out')

        # Check shapes are okay
        self.assert_dmri_input(dmri_in)
        self.assert_vec_input(bvec_in)
        self.assert_vec_output(bvec_out)

        data = tf.data.Dataset.from_tensor_slices(((dmri_in, bvec_in, bvec_out),))
        data = data.batch(self._config['batch_size'])

        # Run model inference
        datasets['dmri_out'] = self.model.predict(data, verbose=1)

    @staticmethod
    def save_dmri_data(datasets, context, fpath):
        '''Saves dmri_out within datasets dict

        Args:
            datasets (Dict[str,Any]):
                'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
            context (Dict[str,Any]):
                'affine': (np.ndarray) -> shape (4, 4)
            fpath (str): Filepath to save
        '''
        save_nifti(datasets['dmri_out'], context['affine'], fpath)


class InferenceProcessorNorm(InferenceProcessor):
    '''Inference Processor with individual xmax normalisation'''

    def __init__(self, model, **kwargs):
        '''Initializes processor object

        Args:
            model (tf.keras.models.Model): Compiled model with weights
                already loaded.
        Keyword Args:
            shell (int): Selected shell
                Default: 1000
            patch_shape_in (Tuple[int,int,int]): shape of input patches to train/infer with
                Default: determined by `model`.
            pcent (int): Maximum percentile normalisation. Default: 99
            batch_size (int): Batch size of dataset for inference/training.
                Default: 4
        '''
        super().__init__(model, **kwargs)
        self.model = model
        self._scaler = ScalerNorm

    def _get_config(self, kwargs):
        '''Sets internal default keyword args `self._config`'''
        config = {
            'shell': kwargs.get('shell', 1000),
            'patch_shape_in': kwargs.get(
                'patch_shape_in', self.model.input[0].shape[-3:]
            ),
            'pcent': kwargs.get('pcent', 99),
            'batch_size': kwargs.get('batch_size', 4),
        }
        return config
