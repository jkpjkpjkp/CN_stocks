import numpy as np

x = np.load('../data/train.npy')

x = x.reshape(-1, 119)

y = np.prod(np.lib.stride_tricks.sliding_window_view(x, (1,30)), axis=-1)
print(y)