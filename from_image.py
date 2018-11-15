import numpy as np
import matplotlib.image as mpimg
from keras.models import load_model
from skimage import io


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


img = io.imread('./data/test.png', as_grey=True)
# img = rgb2gray(img)
# mpimg.imsave('test.thumbnail.png', gray)

X_test = np.array(img).astype('float32') / 255

model = load_model('network.h5')
print(X_test.reshape(1, 28 * 28))
print(model.predict_proba(X_test.reshape(1, 28 * 28)))
print(model.predict_classes(X_test.reshape(1, 28 * 28)))
