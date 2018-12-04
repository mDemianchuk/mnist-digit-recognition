import base64
import cv2
import math
import os
from io import BytesIO

import numpy as np
from PIL import Image, ImageChops, ImageOps
from scipy import ndimage
from skimage import io

from constants import RGB_MAX, IMG_EXTENSION, IMG_SIZE, INNER_BOX, OUTER_BOX, IMG_FORMAT, SEQ_MODEL_PATH, \
    LR_MODEL_PATH, ALPHA


def image_to_shape(img):
    img = np.array(img).astype('float32')
    img = img / RGB_MAX
    img = img.reshape(1, IMG_SIZE)
    return img


def import_images_from_dir(path, model):
    image_list = []
    label_list = []
    directory = os.fsencode(path)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(IMG_EXTENSION):
            image = io.imread(path + filename)
            image_list.append(image)
            label = int(filename[0])
            if model == SEQ_MODEL_PATH:
                label_arr = np.zeros(10)
                label_arr[label] = 1
                label_list.append(label_arr)
            elif model == LR_MODEL_PATH:
                label_list.append(int(filename[0]))
        else:
            continue
    return np.array(image_list), np.array(label_list)


def normalize(dataset):
    dataset = dataset.reshape((dataset.shape[0], IMG_SIZE))
    dataset = dataset.astype('float32') / RGB_MAX
    return dataset


def resize(img_arr):
    rows, cols = img_arr.shape
    if rows > cols:
        factor = float(INNER_BOX) / rows
        rows = INNER_BOX
        cols = int(round(cols * factor))
    else:
        factor = float(INNER_BOX) / cols
        cols = INNER_BOX
        rows = int(round(rows * factor))

    # Fits a digit into a 20x20 bounding box
    img_arr = cv2.resize(img_arr, (cols, rows))

    # Adds missing rows & columns to get a 28x28 image
    cols_padding = (int(math.ceil((OUTER_BOX - cols) / 2.0)), int(math.floor((OUTER_BOX - cols) / 2.0)))
    rows_padding = (int(math.ceil((OUTER_BOX - rows) / 2.0)), int(math.floor((OUTER_BOX - rows) / 2.0)))
    img_arr = np.lib.pad(img_arr, (rows_padding, cols_padding), 'constant')

    return img_arr


def getBestShift(img_arr):
    rows, cols = img_arr.shape
    cy, cx = ndimage.measurements.center_of_mass(img_arr)
    shift_x = np.round(cols / 2.0 - cx).astype(int)
    shift_y = np.round(rows / 2.0 - cy).astype(int)
    return shift_x, shift_y


def shift(img_arr, sx, sy):
    rows, cols = img_arr.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted_img_arr = cv2.warpAffine(img_arr, M, (cols, rows))
    return shifted_img_arr


def trim(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return img.crop(bbox)


def remove_transparency(img, bg_color=(RGB_MAX, RGB_MAX, RGB_MAX)):
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = img.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new('RGB', img.size, bg_color + (ALPHA,))
        bg.paste(img, mask=alpha)
        return bg

    else:
        return img


def full_conversion(img):
    img = remove_transparency(img)
    img = img.convert('L')
    img = trim(img)
    img = img.crop()
    img = ImageOps.invert(img)

    # Need to have access to pixels by the index
    img_arr = np.asarray(img)
    img_arr = resize(img_arr)
    # Shifts an image using the center of mass
    shift_x, shift_y = getBestShift(img_arr)
    img_arr = shift(img_arr, shift_x, shift_y)

    # Converts an array back to PIL Image
    img = Image.fromarray(img_arr)
    return img


def image_from_b64(data):
    img = Image.open(BytesIO(base64.urlsafe_b64decode(data)))
    img = full_conversion(img)
    return img


def convert_b64_image(data):
    img = image_from_b64(data)
    buffered = BytesIO()
    img.save(buffered, format=IMG_FORMAT)
    return base64.b64encode(buffered.getvalue())


def default(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError
