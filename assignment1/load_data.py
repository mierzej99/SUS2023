import skimage
import numpy as np


def load_images(file_with_paths):
    data = []
    file_names = []
    with open(file_with_paths, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        img = skimage.io.imread(line)
        img = skimage.transform.resize(img, (15, 15, 3))
        data.append(img.flatten())
        file_names.append(line)
    data = np.asarray(data)
    return data, file_names
