from keras.datasets import mnist
import keras
from tensorflow import nn

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

network = keras.models.Sequential([
    keras.layers.Dense(512, activation=nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=nn.softmax)
])

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=10, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

network.save('network.h5')  # creates a HDF5 file 'my_model.h5'

print('test_acc:', test_acc)
