import numpy as np
from torch.utils.data import Dataset


class quantile_1min(Dataset):
    def __init__(self, config, filename='../data/train.npy'):
        super().__init__()
        self.data = np.load(filename)
        self.q = np.load('./.results/128th_quantiles_of_1min_ret.npy')
        self.seq_len=config.seq_len
        assert self.data.shape[0] % self.seq_len == 0

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx * self.seq_len : (idx+1) * self.seq_len]
        x = np.searchsorted(self.q, x)
        return x

class quantile_30min(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.data = np.load(filename) if isinstance(filename, str) else filename
        self.q = np.load('./.results/128th_quantiles_of_1min_ret.npy')
        self.q30 = np.load('./.results/q30.npy')
    
    def __len__(self):
        return len(self.data) // 119

    def __getitem__(self, idx):
        x = self.data[idx * 119 : (idx+1) * 119]
        y = np.prod(np.lib.stride_tricks.sliding_window_view(x, 30), axis=-1).flatten()
        y = np.searchsorted(self.q30, y)
        x = np.searchsorted(self.q, x)
        return x, y