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
        if img.shape[2] == 3:
            img = skimage.color.rgb2gray(img)
        else:
            img = skimage.color.rgba2rgb(img)
            img = skimage.color.rgb2gray(img)
        img = skimage.transform.resize(img, (15, 14))
        data.append(img.flatten())
        file_names.append(line)
    data = np.asarray(data)
    return data, file_names
