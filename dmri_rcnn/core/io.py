'''I/O operations'''

import numpy as np
import nibabel as nib


def load_bvec(fpath):
    '''Loads bvec into numpy array

    Args:
        fpath (str): path to bvec file

    Returns:
        bvec (np.ndarray): bvec array, shape -> (3, b)
    '''
    bvec = np.genfromtxt(fpath, dtype=np.float32)
    if bvec.shape[1] == 3:
        bvec = bvec.T

    return bvec


def load_bval(fpath):
    '''Loads bval into numpy array

    Args:
        fpath (str): path to bvec file

    Returns:
        bval (np.ndarray): bval array, shape -> (b,)
    '''
    return np.genfromtxt(fpath, dtype=np.float32)


def load_nifti(nifti_fpath, dtype=np.float32, force_ras=False):
    '''Loads NIfTI image into memory

    Args:
        nifti_fpath (str): Filepath to nifti image
        dtype (type): Datatype to load array with.
            Default: `np.float32`
        force_ras (bool): Forces data into RAS data ordering scheme.
            Default: `False`.

    Returns:
        data (np.ndarray): image data
        affine (np.ndarray): affine transformation -> shape (4, 4)
    '''
    img = nib.load(nifti_fpath)
    if force_ras:
        if nib.aff2axcodes(img.affine) != ('R', 'A', 'S'):
            print(f'Converting {img.get_filename()} to RAS co-ords')
            img = nib.as_closest_canonical(img)
    data = np.asarray(img.dataobj, dtype=dtype)

    return data, img.affine


def save_nifti(data, affine, fpath, descrip=None):
    '''Saves NIfTI image to disk.

    Args:
        data (np.ndarray): Data array
        affine (np.ndarray): affine transformation -> shape (4, 4)
        fpath (str): Filepath to save to.
        descrip (str): Additional info to add to header description
            Default: `None`.
    '''
    img = nib.Nifti2Image(data, affine)

    if descrip is not None:
        img.header['descrip'] = descrip

    nib.save(img, fpath)


def save_bvec(bvec, fpath):
    '''Saves bvec to file in shape (3, b)
            'bval': (np.ndarray) -> shape (b,)
    Args:
        bvec (np.ndarray): bvec array, accepts shapes
            (3, b) or (b, 3).
        fpath (str): filepath to save bvec to.
    '''
    if bvec.shape[1] == 3:
        bvec = bvec.T

    np.savetxt(fpath, bvec, fmt='%1.6f')


def save_bval(bval, fpath):
    '''Saves bval to file

    Args:
        bval (np.ndarray): bval array shape -> (b,)
        fpath (str): filepath to save bval to
    '''
    np.savetxt(fpath, bval, newline=' ', fmt='%g')
