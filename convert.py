import math
import os

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps
from scipy import ndimage


def resize(img_arr):
    rows, cols = img_arr.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))

    # Fits a digit into a 20x20 bounding box
    img_arr = cv2.resize(img_arr, (cols, rows))

    # Adds missing rows & columns to get a 28x28 image
    cols_padding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rows_padding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
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


def remove_transparency(img, bg_colour=(255, 255, 255)):
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = img.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGB", img.size, bg_colour + (255,))
        bg.paste(img, mask=alpha)
        return bg

    else:
        return img


input = './data/initial/'
output = './data/converted/'
directory = os.fsencode(input)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        img = Image.open(input + filename)
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
        img.save(output + filename, "PNG")
        continue
    else:
        continue
