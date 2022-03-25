'''Function to get weights path'''
import os

def get_weights(model_dim: int, shell: int, q_in: int, combined: bool = False):
    '''Gets weights given model parameters
    
    Args:
        model_dim (int): Model dimensionality, either 1 or 3
        shell (int): dMRI shell
        q_in (int): Number of input q-space samples
        combined (bool): Return combined model if available
    
    Returns:
        (str): Weight path to be used in model.load_weights method.
            Will return `None` if directory not found.
    '''
    root_dir = os.path.dirname(__file__)
    if combined:
        weight_dir = os.path.join(root_dir, f'{model_dim}D_RCNN', f'{q_in}in')
    else:
        weight_dir = os.path.join(root_dir, f'{model_dim}D_RCNN', f'b{shell}_{q_in}in')
    if os.path.isdir(weight_dir):
        return os.path.join(weight_dir, 'weights')
    return None
