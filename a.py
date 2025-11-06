import os
import h5py
import numpy as np

def get_file(file):
    with h5py.File(file, 'r') as f:
        return np.array(f['values'])

data = []
from os.path import join, getsize
for root, dirs, files in os.walk('/data/share/data'):
    for file in files:
        if ('_' not in file or file[-5] == '_' and file[-4] == '0') and getsize(join(root, file)) == 163213752:
            data.append(get_file(join(root, file)))

data = np.concat(data)

print(data.shape)



