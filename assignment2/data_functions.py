import pathlib
import skimage
from tqdm import tqdm
import numpy as np
import pandas as pd


def list_of_paths(path):
    # creating lists of paths to photos and encodings

    df = pd.read_csv(path, header=0, usecols=[1, 2])
    df = df.applymap(lambda x: x.replace('BigDataCup2022/S1', 'data'))
    input_list = list(df['input_path'])
    enc_list = list(df['encoded_path'])
    return input_list, enc_list


import random


def create_list_of_pairs_and_labels(input_list, enc_list, number_of_not_correct_pairs):
    step = number_of_not_correct_pairs + 1
    n_data = step * len(input_list)
    pairs = [0] * n_data
    labels = [0] * n_data
    for photo_idx in range(0, len(input_list)):
        pairs[photo_idx * step] = [input_list[photo_idx], enc_list[photo_idx]]
        labels[photo_idx * step] = 1  # pair of photo and its encoding
        enc_without_correct_pair = [x for i, x in enumerate(enc_list) if i != photo_idx]
        for pair_idx in range(1, step):
            enc = random.choice(enc_without_correct_pair)
            enc_without_correct_pair.remove(enc)
            pairs[photo_idx * step + pair_idx] = [input_list[photo_idx], enc]
            labels[photo_idx * step + pair_idx] = 0  # pair of photo and encoding of some other photo
    return pairs, labels


def load_images_scikit(files, gray_scale=True, size=64):
    """
    Takes file with paths to input images and load then and their names to arrays
    """
    images = []
    for inp, enc in tqdm(files):
        # loading images
        img_org = skimage.io.imread(inp)
        img_enc = skimage.io.imread(enc)
        # changing to gray_scale
        if gray_scale:
            img_org = skimage.color.rgb2gray(img_org)
            img_enc = skimage.color.rgb2gray(img_enc)
        # resizing images to 15X13
        img_org = skimage.transform.resize(img_org, (size, size))
        img_enc = skimage.transform.resize(img_enc, (size, size))
        # flattening images to vectors in row manner
        img_org = img_org.flatten(order='C')
        img_enc = img_enc.flatten(order='C')

        final_data = np.concatenate((img_org, img_enc))
        images.append(final_data)
        # appending file and its name to appropriate arrays
        # np.array from array
    images = np.asarray(images)
    return images


def load_and_transform_images_scikit(files, gray_scale=True, size=64):
    """
    Takes file with paths to input images and load then and their names to arrays
    """
    images = []
    for inp, enc in tqdm(files):
        # loading images
        img_org = skimage.io.imread(inp)
        img_enc = skimage.io.imread(enc)
        # changing to gray_scale
        if gray_scale:
            img_org = skimage.color.rgb2gray(img_org)
            img_enc = skimage.color.rgb2gray(img_enc)
        # resizing images to 15X13
        img_org = skimage.transform.resize(img_org, (size, size))
        img_enc = skimage.transform.resize(img_enc, (size, size))
        # flattening images to vectors in row manner
        img_org = img_org.flatten(order='C')
        img_enc = img_enc.flatten(order='C')

        final_data = np.concatenate((img_org, img_enc))

        # appending file and its name to appropriate arrays
        # np.array from array
        images.append(final_data)
    images = np.asarray(images)
    dataset = files[0][0].split('/')[1]
    if gray_scale:
        gs = '_gray'
    else:
        gs = ''
    file_name = f'transformed_data/{dataset}_{size}{gs}_{len(files)}.csv'
    np.savetxt(file_name, images)
