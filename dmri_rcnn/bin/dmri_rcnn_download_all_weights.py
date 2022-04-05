#!/usr/bin/env python3
'''Script to Download all dMRI RCNN weights'''

from dmri_rcnn.core.weights import get_weights


if __name__ == '__main__':
    for model_dim in (1, 3):
        for shell in (1000, 2000, 3000, 'all'):
            for q_in in (6, 10, 30):
                try:
                    get_weights(model_dim, shell, q_in)
                except AttributeError:
                    pass
