import torch.nn as nn

class Model(nn.Module):
    def __init__(self, module_name):
        super(Model, self).__init__()
        