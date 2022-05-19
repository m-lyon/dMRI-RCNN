#!/usr/bin/env python3
'''Script to run dMRI RCNN inference'''
import os
import argparse

from dmri_rcnn.core.processing.sph_harmonic import SphericalHarmonicProcessor


def fpath(string: str):
    '''Checks filepath exists'''
    if os.path.isfile(string):
        return string
    raise RuntimeError(f'Filepath {string} does not exist.')


def main(args: argparse.ArgumentParser):
    '''Kicks off main script'''
    processor = SphericalHarmonicProcessor(args.shell)
    processor.run_subject(args.dmri_in, args.bvec_in, args.bvec_out, args.dmri_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('dMRI Spherical Harmonic Baseline Inference')
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
        required=True,
        help='Shell to perform inference on. Must be same shell as context/target dMRI and b-vecs',
    )

    arguments = parser.parse_args()
    main(arguments)
