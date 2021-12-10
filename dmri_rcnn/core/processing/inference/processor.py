'''Inference processor class'''
import tensorflow as tf

from ...io import load_raw_data, save_nifti

from .scaler import Scaler
from .patcher import Patcher
from .splitter import Splitter

class InferenceProcessor:
    '''IntraShell pipeline with V3Scaler'''

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
            norms (Dict[int,Tuple[float,float]]): Normalisations for each shell
                Default: {1000: 4000.0, 2000: 3000.0, 3000: 2000.0}
            batch_size (int): Batch size of dataset for inference/training.
                Default: 52
        '''
        self.model = model
        self._scaler = Scaler
        self._patcher = Patcher
        self._splitter = Splitter
        self._config = self._get_config(kwargs)

    def _get_config(self, kwargs):
        '''Sets internal default keyword args `self._config`'''
        config = {
            'shell': kwargs.get('shell', 1000),
            'patch_shape_in': kwargs.get('patch_shape_in', self.model.input[0].shape[-3:]),
            'norms': kwargs.get('norm_val', {1000: 4000.0, 2000: 3000.0, 3000: 2000.0}),
            'batch_size': kwargs.get('batch_size', 52)
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
        for key in self._config:
            print('    {} -> {}'.format(key, self._config[key]))

    def load_raw_data_dict(self, dmri_in_fpath, bvec_in_fpath, bvec_out_fpath, mask_fpath):
        '''Loads dMRI data into memory and zips to datasets dict

        Args:
            dmri_in_fpath (str): Path to input dMRI data file.
            bvec_in_fpath (str): Path to input b-vector data file.
            bvec_out_fpath (str): Path to output b-vector data file.
            mask_fpath (str): Path to brain mask file.

        Returns:
            datasets (Dict[str,Any]):
                'mask': (np.ndarray) -> shape (i, j, k)
                'dmri_in': (np.ndarray) -> shape (i, j, k, q_in)
                'bvec_in': (np.ndarray) -> shape (3, q_in)
                'bvec_out': (np.ndarray) -> shape (3, q_out)
            context (Dict[str,Any]):
                'affine': (np.ndarray) -> shape (4, 4)
        '''
        dmri_in, bvec_in, bvec_out, mask, affine = load_raw_data(
            dmri_in_fpath, bvec_in_fpath, bvec_out_fpath, mask_fpath
        )

        datasets = {
            'mask': mask,
            'dmri_in': dmri_in,
            'bvec_in': bvec_in,
            'bvec_out': bvec_out,
        }

        context = {'affine': affine}

        return datasets, context

    def run_subject(self, dmri_in, bvec_in, bvec_out, mask, dmri_out=None):
        # TODO: docstring
        datasets, context = self.load_raw_data_dict(dmri_in, bvec_in, bvec_out, mask)
        self.preprocess(datasets, context)
        self.run_model(datasets)
        self.postprocess(datasets)
        if dmri_out is not None:
            self.save_dmri_data(datasets, context)

        return datasets['dmri_out']

    def preprocess(self, datasets, context):
        # TODO: docstring
        # Scales data to approximate range [0,1]
        self._scaler.forward(datasets, context, **self._config)
        # Splits dMRI data into 3D patches
        self._patcher.forward(datasets, context, **self._config)
        # Reshapes and repeats dataset
        self._splitter.forward(datasets)

    def postprocess(self, datasets, context):
        # TODO: docstring
        # Reshape dMRI
        self._splitter.backward(datasets)
        # Combines 3D patches to contiguous volume
        self._patcher.backward(datasets, context)
        # Rescale data
        self._scaler.backward(datasets, context)

    def run_model(self, datasets):
        '''Runs model through inference to produce dMRI outputs

        Args:
            datasets (Dict[str,Any]):
                'bvec_in': (np.ndarray) -> shape (X, q_in, 3)
                'bvec_out': (np.ndarray) -> shape (X, q_out, 3)
                ...

        Modifies:
            datasets (Dict[str,Any]):
                + 'dmri_out': (np.ndarray) -> shape (X, q_out, n, m, o)
                - 'dmri_in': (np.ndarray) -> shape (X, q_in, m, n, o)
                - 'bvec_in': (np.ndarray) -> shape (X, q_in, 3)
                - 'bvec_out': (np.ndarray) -> shape (X, q_out, 3)
                ...
        '''
        # pylint: disable=unused-argument
        print('Running inference on data...')

        dmri_in = datasets.pop('dmri_in')
        bvec_in = datasets.pop('bvec_in')
        bvec_out = datasets.pop('bvec_out')

        # Check shapes are okay
        self.assert_dmri_input(dmri_in)
        self.assert_vec_input(bvec_in)
        self.assert_vec_output(bvec_out)

        data = tf.data.Dataset.from_tensor_slices(((dmri_in, bvec_in, bvec_out), ))
        data = data.batch(self._config['batch_size'])

        # Run model inference
        datasets['dmri_out'] = self.model.predict(data, verbose=1)

    def save_dmri_data(self, datasets, context, fpath):
        '''Saves dmri_out within datasets dict
        
        Args:
            datasets (Dict[str,Any]):
                'dmri_out': (np.ndarray) -> shape (i, j, k, q_out)
            context (Dict[str,Any]):
                'affine': (np.ndarray) -> shape (4, 4)
            fpath (str): Filepath to save 
        '''
        save_nifti(datasets['dmri_out'], context['affine'], fpath)
