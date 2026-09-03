import torch
from torchmfbd.regularization import Regularization

class RegularizationTime(Regularization):

    def __init__(self, lambda_reg=0.0, variable=None):
        super().__init__('time', variable, lambda_reg)
        
        self.lambda_reg = lambda_reg

    def __call__(self, x):
                           
        return 0.0