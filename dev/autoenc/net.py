import torch
from torch import nn
from torch.nn import functional as F

class Enc(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.num_stocks, config.hidden_dim)
        self.l21 = nn.Linear(config.num_stocks, config.intermediate_dim)
        self.l22 = nn.Linear(config.intermediate_dim, config.hidden_dim)
        self.l3 = nn.Linear(config.hidden_dim, config.bottleneck_dim)
        self.l41 = nn.Linear(config.bottleneck_dim, config.intermediate_dim)
        self.l42 = nn.Linear(config.intermediate_dim, config.bottleneck_dim)
    
    def forward(self, x):
        x = self.l1(x) + self.l22(F.silu(self.l21(x)))
        x = F.silu(x)
        return self.l3(x) + self.l42(F.silu(self.l41(x)))