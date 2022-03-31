''' Stream I/O functions'''
import numpy as np
import tensorflow as tf


def _tensor_bytes_feature(tensor):
    '''Returns a bytes_list from a Tensor.'''
    str_bytes = tf.io.serialize_tensor(tensor).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str_bytes]))


def _int64_feature(value):
    '''Returns an int64_list from a bool / enum / int / uint.'''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _serialize_dataset(dmri, bvec, bval, mask):
    '''Serializes a dataset (individual example)

    Args:
        dmri (np.ndarray): shape -> (i, j, k, b)
        bvec (np.ndarray): shape -> (3, b)
        bval (np.ndarray): shape -> (b,)
        mask (np.ndarray): shape -> (i, j, k)

    Returns:
        example_proto (tf.train.Example): an Example dataset
    '''
    feature = {
        'dmri': _tensor_bytes_feature(dmri),
        'bvec': _tensor_bytes_feature(bvec),
        'bval': _tensor_bytes_feature(bval),
        'mask': _tensor_bytes_feature(mask.astype(np.int8)),
        'i': _int64_feature(dmri.shape[0]),
        'j': _int64_feature(dmri.shape[1]),
        'k': _int64_feature(dmri.shape[2]),
        'b': _int64_feature(dmri.shape[3]),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto


def _parse_dataset_element(example_proto):
    '''Parses serialized dataset into Tensors

    Args:
        example_proto (tf.train.Example): an Example dataset

    Returns:
        data_out (Dict[str,tf.Tensor]) dataset example in tf.Tensor format
            'dmri': (tf.Tensor) shape -> (i, j, k, b)
            'bvec': (tf.Tensor) shape -> (3, b)
            'bval': (tf.Tensor) shape -> (b,)
            'mask': (tf.Tensor) shape -> (i, j, k)
    '''
    feature_description = {
        'i': tf.io.FixedLenFeature([], tf.int64),
        'j': tf.io.FixedLenFeature([], tf.int64),
        'k': tf.io.FixedLenFeature([], tf.int64),
        'b': tf.io.FixedLenFeature([], tf.int64),
        'dmri': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'bvec': tf.io.FixedLenFeature([], tf.string),
        'bval': tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(example_proto, feature_description)

    i, j, k, b = content['i'], content['j'], content['k'], content['b']
    dmri, mask = content['dmri'], content['mask']
    bvec, bval = content['bvec'], content['bval']

    dmri = tf.io.parse_tensor(dmri, out_type=tf.float32)
    dmri = tf.reshape(dmri, shape=[i, j, k, b])
    mask = tf.io.parse_tensor(mask, out_type=tf.int8)
    mask = tf.reshape(mask, shape=[i, j, k])
    bvec = tf.io.parse_tensor(bvec, out_type=tf.float32)
    bvec = tf.reshape(bvec, shape=[3, b])
    bval = tf.io.parse_tensor(bval, out_type=tf.float32)
    bval = tf.reshape(bval, shape=[b])

    data_out = {'dmri': dmri, 'bvec': bvec, 'bval': bval, 'mask': mask}

    return data_out


def save_tfrecord_data(dmri, bvec, bval, mask, fpath):
    '''Serializes and saves data in tfrecord format. Optionally
        provide xmax data for dataset.

    Args:
        dmri (np.ndarray): shape -> (i, j, k, b)
        bvec (np.ndarray): shape -> (3, b)
        bval (np.ndarray): shape -> (b,)
        mask (np.ndarray): shape -> (i, j, k)
        fpath (str): Filepath to save data to.
    '''
    proto = _serialize_dataset(dmri, bvec, bval, mask)
    with tf.io.TFRecordWriter(fpath) as writer:
        writer.write(proto.SerializeToString())


def load_tfrecord_data(data_fpaths, run_par=True):
    '''Loads tfrecord datasets object
    (actual dataset loading into memory is done later)

    Args:
        data_fpaths (List[str,]): List of dataset files.
        run_par (bool): Load data in parallel.

    Returns:
        (tf.data.Dataset): Dataset object
    '''
    raw_data = tf.data.TFRecordDataset(data_fpaths)
    num_pcalls = tf.data.AUTOTUNE if run_par else None
    return raw_data.map(_parse_dataset_element, num_parallel_calls=num_pcalls)
