import os

from constants import CONVERTED_PATH, INITIAL_PATH, IMG_EXTENSION, IMG_FORMAT
from tools import full_conversion
from PIL import Image


directory = os.fsencode(INITIAL_PATH)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(IMG_EXTENSION):
        img = Image.open(INITIAL_PATH + filename)
        img = full_conversion(img)
        img.save(CONVERTED_PATH + filename, IMG_FORMAT)
