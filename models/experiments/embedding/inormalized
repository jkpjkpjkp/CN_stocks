import polars as pl
import torch
from torch import nn
from torch.nn import functional as F, Module
from models.prelude.model import dummyLightning, transformerConfig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms


class res_(dummyLightning):
    """
    ! not normalized enc (log scale)
    """
    def __init__(self, config):
        super().__init__(config)
        self.straight = nn.Linear(config.window_size, config.hidden_dim)
        self.l1 = nn.Linear(config.window_size, config.intermediate_dim)
        self.l2 = nn.Linear(config.intermediate_dim, config.hidden_dim)

    def forward(self, x):
        x = x.unflatten(-1, (-1, self.config.window_size))
        a, b = torch.chunk(self.l1(x), 2, dim=-1)
        a = F.relu(a)
        b = F.silu(b)
        x = self.straight(x) + self.l2(torch.concat((a, b), dim=-1))
        return x
