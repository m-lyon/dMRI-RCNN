'''tf.Graph processor used for training'''

from .shell_filter import ShellFilter
from .shell_reorder import ShellReorder
from .scaler import DataScaler, NormDataScaler
from .patcher import Patcher
from .reshaper import Reshaper, Joiner
from .base import CombineDatasets
from .io import load_tfrecord_data


class TrainingProcessor:
    '''Training pipeline

    Applies following steps:
        1) Filter out unwanted shells, ie. those != `shell`
        2) Re-orders qspace into near optimal given set sizes
            `q_in` and `q_out`
        3) Rescales data to given an `xmax`
        4) Splits into 3D patches of size `patch_shape`
        5) Splits into input & output sets of size `q_in` and `q_out`
        6) Flattens subject dimension into patch dimension.

    Output Specs:
        (dmri_in, bvec_in, bvec_out), dmri_out)
            dmri_in (tf.Tensor): shape -> (q_in, m, n, o)
            bvec_in (tf.Tensor): shape -> (q_in, 3)
            bvec_out (tf.Tensor): shape -> (q_out, 3)
            dmri_out (tf.Tensor): shape -> (q_out, m, n, o)
    '''

    def __init__(self, shells, q_in, q_out=10, patch_shape=(10, 10, 10), **kwargs):
        '''InterShell Pipeline

        Args:
            shells (List[int,]): Shells to use in training.
            q_in (int): Number of qspace samples per input
            q_out (int): Number of qspace samples per output.
                Default: 10
            patch_shape (Tuple[int,int,int]): 3D patch shape for input & output.
                Default: (10, 10, 10)

        Keyword Args:
            norms (Dict[int,Tuple[float,float]]): Normalisations for each shell
                {`shell`: xmax}. Default: {1000: 4000., 2000: 3000., 3000: 2000.}
            shell_var (float): Shell variance for shell group membership filter.
                Default: 30.0
            random_seed (bool): Use random seed when selecting optimal subsets
                Default: True
        '''
        seed = kwargs.get('random_seed', True)
        shell_var = kwargs.get('shell_var', 30.0)
        norms = kwargs.get('norms', {1000: 4000.0, 2000: 3000.0, 3000: 2000.0})

        self.shell_filter = ShellFilter(shells, shell_var=shell_var)
        self.shell_reorder = ShellReorder(q_in, q_out, random_seed=seed)
        self.scaler = DataScaler(norms)
        self.patcher = Patcher(patch_shape)
        self.reshaper = Reshaper(q_in, q_out)
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

    def load_data(self, fpaths, bsize=4, run_par=True, validation=False, buffer_size=10000):
        '''Creates pre-processing pipeline, ready for training

        Args:
            fpaths (List[str, ...]): List of .tfrecord filepaths containing
                training data examples
            bsize (int): Batch size of training examples. Lower this if experiencing
                GPU OOM problems.
            run_par (bool): Run data loading in parallel. Will use all available
                CPUs. Default: `True`
            validation (bool): Designate as validation dataset. Will not apply
                shuffling and shell reordering. Default: `False`
            buffer_size (int): Buffer size used for shuffled dataset, therefore only
                applicable if validation == `False`. Lower this number if experiencing
                CPU RAM OOM problems. Default: 10000

        Returns:
            datasets (tf.data.Dataset): Dataset with preprocessing mapping.
        '''
        dataset = load_tfrecord_data(fpaths)
        dataset = self(dataset, run_par=run_par, validation=validation)
        if not validation:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size=bsize)

        return dataset


class TrainingProcessorNorm(TrainingProcessor):
    '''Training Processor with Percentile normalisation'''

    def __init__(self, shells, q_in, q_out=10, patch_shape=(10, 10, 10), **kwargs):
        '''InterShell Pipeline

        Args:
            shells (List[int,]): Shells to use in training.
            q_in (int): Number of qspace samples per input
            q_out (int): Number of qspace samples per output.
                Default: 10
            patch_shape (Tuple[int,int,int]): 3D patch shape for input & output.
                Default: (10, 10, 10)

        Keyword Args:
            pcent (int): Percentile normalisation. Default: 99
            shell_var (float): Shell variance for shell group membership filter.
                Default: 30.0
            random_seed (bool): Use random seed when selecting optimal subsets
                Default: True
        '''
        # pylint: disable=super-init-not-called
        seed = kwargs.get('random_seed', True)
        shell_var = kwargs.get('shell_var', 30.0)
        pcent = kwargs.get('pcent', 99)

        self.shell_filter = ShellFilter(shells, shell_var=shell_var)
        self.shell_reorder = ShellReorder(q_in, q_out, random_seed=seed)
        self.scaler = NormDataScaler(pcent)
        self.patcher = Patcher(patch_shape)
        self.reshaper = Reshaper(q_in, q_out)
        self.joiner = Joiner()
        self.combiner = CombineDatasets()
