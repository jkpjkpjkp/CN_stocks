import numpy as np

class halfdayData:
    def __init__(self, filename='../data/train.npy'):
        self.data = np.load(filename) if isinstance(filename, str) else filename
        self.seq_len = 119
    def __len__(self):
        return len(self.data) // self.seq_len
    def __getitem__(self, idx):
        return self.data[idx * self.seq_len : (idx+1) * self.seq_len]
