import math
import os, sys
from PIL import Image
import cv2
import matplotlib.image as mpimg
import numpy as np

from PIL import Image, ImageChops


def black_background_thumbnail(source_image, thumbnail_size=(200, 200)):
    background = Image.new('RGBA', thumbnail_size, "white")
    source_image.thumbnail(thumbnail_size)
    (w, h) = source_image.size
    x = math.floor(((thumbnail_size[0] - w) / 2))
    y = math.floor((thumbnail_size[1] - h) / 2)
    background.paste(source_image, (x, y))
    return background


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    # left, upper, right, and lower
    if bbox:
        return im.crop(bbox)


def remove_transparency(im, bg_colour=(255, 255, 255)):
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im


def black_white(col):
    gray = col.convert('L')

    # Let numpy do the heavy lifting for converting pixels to pure black or white
    bw = np.asarray(gray).copy()

    # Pixel range is 0...255, 256/2 = 128
    bw[bw < 128] = 0  # Black
    bw[bw >= 128] = 255  # White
    # bw[bw == 255] = 1
    # bw[bw == 0] = 255
    # bw[bw == 1] = 0

    # Now we put it back in Pillow/PIL land
    imfile = Image.fromarray(bw)
    return imfile



im = Image.open("./data/1_1.png")
im = trim(im)
im = black_background_thumbnail(im)
im = remove_transparency(im)

size = 28, 28
outfile = './data/test.png'
im.thumbnail(size, Image.ANTIALIAS)
# im = black_white(im)
im.save(outfile, "PNG")
