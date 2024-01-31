import torch
from torch import nn
import os
import random
import sys
import numpy as np


def cat2int(x):
    if x == "negative": return 0
    elif x == "neutral": return 1
    else: return 2

def int2cat(x):
    if x == 0: return "negative"
    elif x == 1: return "neutral"
    else: return "positive"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

