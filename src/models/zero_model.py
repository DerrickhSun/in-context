import torch
from core import ContextModel

class ZeroModel(ContextModel):
    def __init__(self, **kwargs):
        super(ZeroModel, self).__init__()
        self.name = "zero_model"
        self.context_length = -1

    def forward(self, xs, ys):
        
        return 0*ys
