import os
import random as rn

import keras
import numpy as np
import tensorflow as tf
from keras import backend as bke
from keras.datasets import mnist
from keras.utils import to_categorical
from tensorflow import nn

from tools import normalize

np.random.seed(1337)
rn.seed(1337)
tf.set_random_seed(1337)

os.environ['TF_CPP_MINLOGLEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'

s = tf.Session(graph=tf.get_default_graph())
bke.set_session(s)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = keras.models.Sequential([
    keras.layers.Dense(512, activation=nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=nn.softmax)
])

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Normalizing the dataset
train_images = normalize(train_images)
test_images = normalize(test_images)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=10, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

network.save('network.h5')  # creates a HDF5 file 'my_model.h5'
print('test_acc:', test_acc)
