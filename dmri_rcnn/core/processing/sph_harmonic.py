'''Spherical Harmonic baseline processor'''

from dipy.core.sphere import Sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf

from ..io import load_nifti, load_bvec, save_nifti


class SphericalHarmonicProcessor:
    '''Spherical Harmonic Baseline Processor'''

    def __init__(self, shell, sh_order=2, basis='tournier07', smooth=0.0):
        '''Initialises SH Processor object

        Args:
            shell (int): Selected shell
            sh_order (int): Spherical Harmonic order.
                Default: 2
            basis (str): Spherical harmonic basis. See dipy.reconst.shm.sf_to_sh
                for more info. Default: "tournier07"
            smooth (float): Lambda regularisation in the SH fit, see
                dipy.reconst.shm.sf_to_sh for more info. Default: 0.0
        '''
        self.shell = shell
        self.sh_order = sh_order
        self.basis = basis
        self.smooth = smooth

    def run_subject(self, dmri_in, bvec_in, bvec_out, dmri_out=None):
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
            recon (np.ndarray): Inferred SH dMRI data
        '''
        # Load data
        dmri, affine = load_nifti(dmri_in)
        bvecs_in = load_bvec(bvec_in)
        bvecs_out = load_bvec(bvec_out)

        sphere_in = Sphere(xyz=bvecs_in.T)
        sphere_out = Sphere(xyz=bvecs_out.T)
        # Get Spherical Harmonic coefficients
        sh_coeffs = sf_to_sh(
            dmri, sphere_in, sh_order=self.sh_order, basis_type=self.basis, smooth=self.smooth
        )
        # Use to infer other directions
        recon = sh_to_sf(sh_coeffs, sphere_out, sh_order=self.sh_order, basis_type=self.basis)

        if dmri_out is not None:
            save_nifti(recon, affine, dmri_out)

        return recon
