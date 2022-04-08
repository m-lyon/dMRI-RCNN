#!/usr/bin/env python3
'''Script to run dMRI RCNN inference'''
import os
import argparse

from dmri_rcnn.core.io import load_bvec
from dmri_rcnn.core.weights import get_weights
from dmri_rcnn.core.model import get_1d_autoencoder, get_3d_autoencoder
from dmri_rcnn.core.processing import InferenceProcessor, InferenceProcessorNorm


def get_q_in(bvec_in, bvec_out):
    '''Infers q-space input dimension from b-vector'''
    bvec_in_arr = load_bvec(bvec_in)
    q_in = bvec_in_arr.shape[-1]

    bvec_out_arr = load_bvec(bvec_out)
    q_out = bvec_out_arr.shape[-1]

    return q_in, q_out


def fpath(string):
    '''Checks filepath exists'''
    if os.path.isfile(string):
        return string
    raise RuntimeError(f'Filepath {string} does not exist.')


def print_args(args):
    '''Prints model parameters'''
    print('Model parameters selected:')
    if args.combined:
        print(f'    Model Type -> {args.model_dim}D Combined')
    else:
        print(f'    Model Type -> {args.model_dim}D')
    print(f'    Shell -> {args.shell}')
    print(f'    q_in -> {args.q_in}')
    print(f'    q_out -> {args.q_out}')


def main(args):
    '''Kicks off main script'''
    args.q_in, args.q_out = get_q_in(args.bvec_in, args.bvec_out)
    print_args(args)
    if args.combined and args.norm:
        weights = get_weights(args.model_dim, 'all_norm', args.q_in)
    elif args.combined:
        weights = get_weights(args.model_dim, 'all', args.q_in)
    else:
        weights = get_weights(args.model_dim, args.shell, args.q_in)

    if args.model_dim == 3:
        model = get_3d_autoencoder(weights)
    else:
        model = get_1d_autoencoder(weights)

    if args.norm:
        processor = InferenceProcessorNorm(model, shell=args.shell, batch_size=args.batch_size)
    else:
        processor = InferenceProcessor(model, shell=args.shell, batch_size=args.batch_size)
    processor.run_subject(
        args.dmri_in, args.bvec_in, args.bvec_out, args.mask, args.dmri_out
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('dMRI RCNN Angular Super-resolution')
    parser.add_argument(
        '-dmri_in',
        dest='dmri_in',
        type=fpath,
        required=True,
        help='Context dMRI NIfTI volume. Must be single-shell and contain q_in 3D volumes',
    )
    parser.add_argument(
        '-bvec_in',
        dest='bvec_in',
        type=fpath,
        required=True,
        help='Context b-vectory text file. Whitespace delimited with 3 rows and q_in columns',
    )
    parser.add_argument(
        '-bvec_out',
        dest='bvec_out',
        type=fpath,
        required=True,
        help='Target b-vector text file. Whitespace delimited with 3 rows and q_out columns',
    )
    parser.add_argument(
        '-mask',
        dest='mask',
        type=fpath,
        required=True,
        help='Brain mask NIfTI volume. Must have space spatial dimensions as dmri_in.',
    )
    parser.add_argument(
        '-dmri_out',
        dest='dmri_out',
        type=str,
        required=True,
        help='Inferred dMRI NIfTI volume. This will contain q_out inferred volumes.',
    )
    parser.add_argument(
        '-s',
        '--shell',
        dest='shell',
        type=int,
        choices=[1000, 2000, 3000],
        required=True,
        help='Shell to perform inference on. Must be same shell as context/target dMRI and b-vecs',
    )
    parser.add_argument(
        '-m',
        '--model-dim',
        dest='model_dim',
        type=int,
        choices=[1, 3],
        default=3,
        help='Model dimensionality, choose either 1 or 3. Default: 3.',
    )
    parser.add_argument(
        '-c',
        '--combined',
        dest='combined',
        action='store_true',
        default=False,
        help='Use combined shell model. Currently only applicable with 3D model and 10 q_in.',
    )
    parser.add_argument(
        '-n',
        '--norm',
        dest='norm',
        action='store_true',
        default=False,
        help='Perform normalisation using 99 percentile of data. '
        + 'Only implemented with --combined flag, and only for q_in = 10',
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        dest='batch_size',
        type=int,
        default=4,
        help='Batch size to run model inference with.',
    )

    arguments = parser.parse_args()
    main(arguments)
