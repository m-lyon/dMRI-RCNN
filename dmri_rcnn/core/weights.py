'''Pretrained weights functions'''

from typing import Union
import os

from urllib.request import urlretrieve
from dataclasses import dataclass

from tqdm import tqdm


LOCAL_DIR = os.environ.get(
    'DMRI_RCNN_DIR', os.path.join(os.path.expanduser('~'), '.dmri_rcnn')
)


@dataclass
class ZenodoPath:
    '''Zenodo path dataclass'''

    record: int
    fname: str

    def __post_init__(self):
        self.root_url = f'https://zenodo.org/record/{self.record}/files'

    @property
    def url(self):
        '''URL of hosted file'''
        return f'{self.root_url}/{self.fname}'


class ZenodoWeight:
    '''Zenodo weight dataclass'''

    def __init__(self, record, num_weights):
        self.index = ZenodoPath(record, 'weights.index')
        self.paths = [self.index]
        for idx in range(num_weights):
            self.paths.append(
                ZenodoPath(record, f'weights.data-{idx:05d}-of-{num_weights:05d}')
            )

    def __iter__(self):
        return iter(self.paths)


WEIGHT_URLS = {
    '1D': {
        1000: {
            6: ZenodoWeight(6397440, 1),
            10: ZenodoWeight(6397460, 2),
            30: ZenodoWeight(6397466, 2),
        },
        2000: {
            10: ZenodoWeight(6397470, 1),
        },
        3000: {
            10: ZenodoWeight(6397476, 1),
        },
    },
    '3D': {
        1000: {
            6: ZenodoWeight(6397318, 2),
            10: ZenodoWeight(6397392, 2),
            30: ZenodoWeight(6397405, 2),
        },
        2000: {
            10: ZenodoWeight(6397413, 2),
        },
        3000: {
            10: ZenodoWeight(6397422, 2),
        },
        'all': {
            10: ZenodoWeight(6397221, 2),
        },
        'all_norm':{
            10: ZenodoWeight(6415106, 2),
        }
    },
}


class DownloadProgressBar(tqdm):
    '''TQDM Download Progress Bar'''

    def update_to(self, b=1, bsize=1, tsize=None):
        '''Progress bar download hook'''
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    '''Downloads url and saves to file with a progress bar'''
    with DownloadProgressBar(
        unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]
    ) as pbar:
        urlretrieve(url, filename=output_path, reporthook=pbar.update_to)


def get_weights(model_dim: int, shell: Union[int, str], q_in: int) -> str:
    '''Gets weights given model parameters, will download if not present.

    Args:
        model_dim: Model dimensionality, either 1 or 3
        shell: dMRI shell, either provide int value or "all" or "all_norm"
            str to get model weights for combined model
        q_in: Number of input q-space samples
        combined: Return combined model if available

    Returns:
        weight_dir: Weight path to be used in model.load_weights method.
            Will raise a RuntimeError if not found.
    '''
    try:
        if shell in {'all', 'all_norm'}:
            weight_dir = os.path.join(LOCAL_DIR, f'{model_dim}D_RCNN', f'{shell}_{q_in}in')
        else:
            weight_dir = os.path.join(
                LOCAL_DIR, f'{model_dim}D_RCNN', f'b{shell}_{q_in}in'
            )
        weight_urls = WEIGHT_URLS[f'{model_dim}D'][shell][q_in]
    except KeyError:
        raise AttributeError(
            'Weights in given configuration not found: '
            + f'{model_dim = }, {shell = }, {q_in = }'
        ) from None

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    for weight in weight_urls:
        if not os.path.exists(os.path.join(weight_dir, weight.fname)):
            download_url(weight.url, os.path.join(weight_dir, weight.fname))

    return os.path.join(weight_dir, 'weights')
