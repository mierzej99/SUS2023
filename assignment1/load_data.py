import skimage
import numpy as np


def load_images(file_with_paths):
    """
    Takes file with absolute paths to input images and load then and their names to arrays
    """
    data = []
    file_names = []
    with open(file_with_paths, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        img = skimage.io.imread(line)
        # changing images to grayscale depending on their format (rgb, rgba)
        if img.shape[2] == 3:
            img = skimage.color.rgb2gray(img)
        else:
            img = skimage.color.rgba2rgb(img)
            img = skimage.color.rgb2gray(img)
        # resizing images to 15X13
        img = skimage.transform.resize(img, (15, 13))
        # flattening images to vectors in row manner
        img = img.flatten(order='C')
        # appending file and its name to appropriate arrays
        data.append(img)
        file_names.append(line)
    # np.array from array
    data = np.asarray(data)
    return data, file_names
