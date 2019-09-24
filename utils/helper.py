import yaml
import numpy as np
import torch
import os

def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config

def init_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True