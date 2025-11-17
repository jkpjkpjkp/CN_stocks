import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, functional as F
import mlflow
import random
from ..prelude.model import dummyLightning
from ..prelude.data import halfdayData

class chrono(dummyLightning):
    def __init__(self, config, trunk):
        super().__init__(config)
        self.trunk = trunk
        self.train_dataset = halfdayData('../data/train.npy')
        self.val_dataset = halfdayData('../data/val.npy')
        
        self.l1 = nn.Conv1d(1, config.intermediate_size, config.patch_size)
        self.l2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.shortcut = nn.Conv1d(1, config.hidden_size, config.patch_size)
        
    def forward(self, x):
        x = self.shortcut(x) + self.l2(F.relu(self.l1(x.unsqueeze(-1)).transpose(1, 2)))
        return x
    
    def step(self, batch):
        pass