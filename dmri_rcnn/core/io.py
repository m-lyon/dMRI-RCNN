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


def autocrop_dmri(dmri, mask):
    '''Crops `dmri` and `mask` data
        so dimensions that contain only zeros are cropped

    Args:
        dmri (np.ndarray): shape -> (i, j, k, b)
        mask (np.ndarray): shape -> (i, j, k)

    Returns:
        dmri (np.ndarray): shape -> (i-ix, k-kx, j-jx, b)
        mask (np.ndarray): shape -> (i-ix, k-kx, j-jx)
    '''

    def _get_data_mask(dmri, mask, axis):
        new_mask = np.expand_dims(mask, axis=-1)
        data_mask = np.concatenate([dmri, new_mask], axis=-1)
        data_mask = np.sum(data_mask, axis=axis).astype(bool)

        return data_mask

    # Axis 0
    data_mask = _get_data_mask(dmri, mask, (1, 2, 3))
    dmri = dmri[data_mask, ...]
    mask = mask[data_mask, ...]

    # Axis 1
    data_mask = _get_data_mask(dmri, mask, (0, 2, 3))
    dmri = dmri[:, data_mask, ...]
    mask = mask[:, data_mask, :]

    # Axis 2
    data_mask = _get_data_mask(dmri, mask, (0, 1, 3))
    dmri = dmri[:, :, data_mask, :]
    mask = mask[:, :, data_mask]

    return dmri, mask


def split_image_to_octants(data):
    '''Splits `data` in half in each
        spatial dimension, yielding 8 patches at an 8th of
        the size.

    Args:
        data (np.ndarray): shape -> (i, j, k, ...)

    Returns:
        klist (List[np.ndarray,]): list of split images
    '''
    i, j, k = data.shape[0], data.shape[1], data.shape[2]

    idx = i // 2

    ilist = []
    ilist.append(data[0:idx, ...])
    ilist.append(data[idx:, ...])

    jdx = j // 2

    jlist = []
    for idata in ilist:
        jlist.append(idata[:, 0:jdx, ...])
        jlist.append(idata[:, jdx:, ...])

    kdx = k // 2

    klist = []
    for jdata in jlist:
        klist.append(jdata[:, :, 0:kdx, ...])
        klist.append(jdata[:, :, kdx:, ...])

    return klist
