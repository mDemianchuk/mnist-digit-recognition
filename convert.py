import os
from tools import full_conversion
from PIL import Image

input = './data/initial/'
output = './data/converted/'
directory = os.fsencode(input)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        img = Image.open(input + filename)
        img = full_conversion(img)
        img.save(output + filename, "PNG")
        continue
    else:
        continue
