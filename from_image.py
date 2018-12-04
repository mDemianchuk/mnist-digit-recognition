from random import randint

import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model

from constants import IMG_EXTENSION, TEST_PATH, SEQ_MODEL_PATH, DIGITS_PATH, OUTER_BOX
from tools import import_images_from_dir
from tools import normalize


def evaluate(X_test, num_of_samples):
    size = X_test.shape[0]
    # If specified a number greater than a size of the test set, displaying all images
    if num_of_samples > size:
        num_of_samples = size

    for i in range(num_of_samples):
        rand_index = randint(0, size-1)
        image = X_test[rand_index]
        test = normalize(image.reshape(1, OUTER_BOX, OUTER_BOX))
        pred = model.predict(test).argmax()
        pred_img = Image.open(DIGITS_PATH + str(pred) + IMG_EXTENSION)

        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(image, cmap='Greys')
        plt.title('Actual')
        plt.axis('off')

        fig.add_subplot(1, 2, 2)
        plt.imshow(pred_img, cmap='Greys')
        plt.title('Predicted')
        plt.axis('off')

        plt.show(block=True)
        plt.show()


def evaluate_all(X_test, y_test):
    X_test = normalize(X_test)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Accuracy is ' + str(test_acc))


model = load_model(SEQ_MODEL_PATH)
X_test, y_test = import_images_from_dir(TEST_PATH, model=SEQ_MODEL_PATH)
evaluate(X_test, num_of_samples=20)
evaluate_all(X_test, y_test)
