import os

import numpy as np
from skimage import io

RESOLUTION = (28 * 28)
RGB_MAX = 255


def import_images_from_dir(path):
    image_list = []
    label_list = []
    directory = os.fsencode(path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            image = io.imread(path + filename)
            image_list.append(image)
            label = int(filename[0])
            label_arr = np.zeros(10)
            label_arr[label] = 1
            label_list.append(label_arr)
        else:
            continue
    return np.array(image_list), np.array(label_list)


def normalize(dataset):
    dataset = dataset.reshape((dataset.shape[0], RESOLUTION))
    dataset = dataset.astype('float32') / RGB_MAX
    return dataset
