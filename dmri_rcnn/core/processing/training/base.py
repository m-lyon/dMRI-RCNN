'''Base classes for tf.Graph processing'''

import tensorflow as tf


class DatasetMapper:
    '''Base Datatset Mapper class'''

    def apply(self, data):
        '''Abstract apply method'''
        raise NotImplementedError

    def __call__(self, dataset, run_par=True):
        num_pcalls = tf.data.AUTOTUNE if run_par else None
        return dataset.map(self.apply, num_parallel_calls=num_pcalls)


class FlatDatasetMapper:
    '''Base Flat Dataset Mapper class'''

    def apply(self, data):
        '''Abstract apply method'''
        raise NotImplementedError

    def __call__(self, dataset):
        return dataset.flat_map(self.apply)


class CombineDatasets(FlatDatasetMapper):
    '''Flattens across patch & set dimension using tf.Graph'''

    def __init__(self):
        '''Flattens across patch & set dimension creating true example datasets

        Call Arguments:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                {
                    'dmri_in': (tf.Tensor),
                    'dmri_out': (tf.Tensor),
                    'bvec_in': (tf.Tensor),
                    'bvec_out': (tf.Tensor),
                }

        Output Spec:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                (
                    (
                        dmri_in (tf.Tensor),
                        bvec_in (tf.Tensor),
                        bvec_out (tf.Tensor),
                    ),
                    dmri_out (tf.Tensor),
                )
        '''

    @tf.function
    def apply(self, data):
        '''Applies algorithm

        Args:
            data (Dict[int,Dict]):
                'dmri_in': (tf.Tensor)
                'dmri_out': (tf.Tensor)
                'bvec_in': (tf.Tensor)
                'bvec_out': (tf.Tensor)

        Returns:
            ((dmri_in, bvec_in, bvec_out), dmri_out)
                dmri_in (tf.Tensor)
                bvec_in (tf.Tensor)
                bvec_out (tf.Tensor)
                dmri_out (tf.Tensor)
        '''
        dmri_in = tf.data.Dataset.from_tensor_slices(data['dmri_in'])
        bvec_in = tf.data.Dataset.from_tensor_slices(data['bvec_in'])
        bvec_out = tf.data.Dataset.from_tensor_slices(data['bvec_out'])
        dmri_out = tf.data.Dataset.from_tensor_slices(data['dmri_out'])

        return tf.data.Dataset.zip(((dmri_in, bvec_in, bvec_out), dmri_out))
