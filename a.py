import os

tot = 0
from os.path import join, getsize
for root, dirs, files in os.walk('/data/share/data'):
    for file in files:
        if ('_' not in file or file[-5] == '_' and file[-4] == '0') and getsize(join(root, file)) == 163213752:
            tot += 1
            print(file)
    # print(root, dirs, files)
    print(root, "consumes", end=" ")
    print(sum(getsize(join(root, name)) for name in files), end=" ")
    print("bytes in", len(files), "non-directory files")
    if '__pycache__' in dirs:
        dirs.remove('__pycache__')  # don't visit __pycache__ directories


print(tot)