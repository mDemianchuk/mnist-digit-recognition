import os
from constants import CONVERTED_PATH, IMG_EXTENSION, IMG_FORMAT, RGB_MAX
from PIL import Image, ImageOps


def merge_images(image1, image2, horizontal=True):
    # image1 = Image.open(file1)
    # image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    if horizontal:
        result_width = width1 + width2
        result_height = max(height1, height2)
    else:
        result_width = max(width1, width2)
        result_height = height1 + height2

    result = Image.new('RGB', (result_width, result_height), (RGB_MAX, RGB_MAX, RGB_MAX, 255))
    if horizontal:
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(width1, 0))
    else:
        result.paste(im=image1, box=(0, 0))
        result.paste(im=image2, box=(0, height1))
    return result


def from_array(arr, horizontal=True):
    current = arr.pop(0)
    while len(arr) > 0:
        current = merge_images(current, arr.pop(0), horizontal)
    return current


previous = None
limit = 1000
n = 0
data = [[], [], [], [], [], [], [], [], [], []]
for file in os.listdir(CONVERTED_PATH):
    if n >= limit:
        break
    n += 1
    filename = os.fsdecode(file)
    if not filename.endswith(IMG_EXTENSION):
        continue

    num = int(filename[0])
    data[num].append(Image.open(CONVERTED_PATH + filename))

data = list(map(lambda x: from_array(x), data))
mosaic = from_array(data, False)
mosaic = ImageOps.invert(mosaic)
mosaic.save('mosaic' + IMG_EXTENSION, IMG_FORMAT)
