import numpy as np
import torch
from torch import nn


def create_model():

    model = nn.Sequential()
    # Linear layer mapping from 784 features, so it should be 784->256->16->10
    # your code here
    # Linear layer mapping from 784 features, so it should be 784->256
    model.add_module('l1', nn.Linear(784, 256, bias=True))
    model.add_module('r1', nn.ReLU())
    # Linear layer mapping from 256 features, so it should be 256->16
    model.add_module('l2', nn.Linear(256, 16, bias=True))
    model.add_module('r2', nn.ReLU())
    # Linear layer mapping from 16 features, so it should be 16->1
    model.add_module('l3', nn.Linear(16, 10, bias=True))
    # model.add_module('r3', nn.ReLU())
    # return model instance (None is just a placeholder)

    return model


def count_parameters(model):
    # your code here
    return sum(p.numel() for p in model.parameters())
