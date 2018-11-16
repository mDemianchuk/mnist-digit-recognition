import os
import numpy as np
from skimage import io


def import_images_from_dir(input):
    allimages = None
    alllabels = None
    directory = os.fsencode(input)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            img = io.imread(input + filename)
            img = np.array(img).astype('float32')
            img = img / 255

            img = img.reshape(1, 28 * 28)

            if allimages is None:
                allimages = img
            else:
                allimages = np.concatenate((allimages, img), axis=0)

            n = int(filename[0])
            label = np.zeros(10).reshape(1, 10)
            label[0, n] = 1

            if alllabels is None:
                alllabels = label
            else:
                alllabels = np.concatenate((alllabels, label), axis=0)

            continue
        else:
            continue
    return allimages, alllabels