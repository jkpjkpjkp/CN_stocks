import numpy as np

class halfdayData:
    def __init__(self, filename='../data/train.npy'):
        self.data = np.load(filename) if isinstance(filename, str) else filename
    def __len__(self):
        return len(self.data) // 119
    def __getitem__(self, idx):
        return self.data[idx * 119 : (idx+1) * 119]
