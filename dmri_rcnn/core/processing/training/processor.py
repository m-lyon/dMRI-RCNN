'''tf.Graph processor used for training'''

from .shell_filter import ShellFilter
from .shell_reorder import ShellReorder
from .scaler import DataScaler
from .patcher import Patcher
from .reshaper import Reshaper, Joiner
from .base import CombineDatasets


class TrainingProcessor:
    '''Training pipeline
    
    Applies following steps:
        1) Filter out unwanted shells, ie. those != `shell`
        2) Re-orders qspace into near optimal given set sizes
            `in_num` and `out_num`
        3) Rescales data to given an `xmax`
        4) Splits into 3D patches of size `patch_shape`
        5) Splits into input & output sets of size `in_num` and `out_num`
        6) Flattens subject dimension into patch dimension.

    Output Specs:
        (dmri_in, bvec_in, bvec_out), dmri_out)
            dmri_in (tf.Tensor): shape -> (in_num, m, n, o)
            bvec_in (tf.Tensor): shape -> (in_num, 3)
            bvec_out (tf.Tensor): shape -> (out_num, 3)
            dmri_out (tf.Tensor): shape -> (out_num, m, n, o)
    '''

    def __init__(self, shells, in_num, out_num, patch_shape, norms, **kwargs):
        '''InterShell Pipeline

        Args:
            shells (List[int,]): Shells to use in training.
            in_num (int): Number of qspace samples per input
            out_num (int): Number of qspace samples per output
            patch_shape (Tuple[int,int,int]): 3D patch shape
                for input & output
            norms (Dict[int,Tuple[float,float]]): Normalisations for each shell
                {`shell`: (xmin, xmax)}

        Keyword Args:
            shell_var (float): Shell variance for shell group membership filter.
                Default: 30.0
            random_seed (bool): Use random seed when selecting optimal subsets
                Default: True
        '''
        seed = kwargs.get('random_seed', True)
        shell_var = kwargs.get('shell_var', 30.0)

        self.shell_filter = ShellFilter(shells, shell_var=shell_var)
        self.shell_reorder = ShellReorder(in_num, out_num, random_seed=seed)
        self.scaler = DataScaler(norms)
        self.patcher = Patcher(patch_shape)
        self.reshaper = Reshaper(in_num, out_num)
        self.joiner = Joiner()
        self.combiner = CombineDatasets()

    def __call__(self, datasets, run_par=True, validation=False):
        if run_par:
            print('Running tf.data.Dataset.map in parallel mode.')

        datasets = self.shell_filter(datasets, run_par)
        if not validation:
            datasets = self.shell_reorder(datasets, run_par)
        datasets = self.scaler(datasets, run_par)
        datasets = self.patcher(datasets, run_par)
        datasets = self.reshaper(datasets, run_par)
        datasets = self.joiner(datasets, run_par)
        datasets = self.combiner(datasets)

        return datasets
