'''Abstract operation class'''


class Operation:
    '''Abstract Operation class'''

    @staticmethod
    def forward(datasets, context, **kwargs):
        '''Abstract forward operation method'''
        raise NotImplementedError

    @staticmethod
    def backward(datasets, context, **kwargs):
        '''Abstract backward operation method'''
        raise NotImplementedError
