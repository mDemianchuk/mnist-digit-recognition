import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model

from tools import import_images_from_dir
from tools import normalize


def evaluate(X_test, num_of_samples):
    if num_of_samples > X_test.shape[0]:
        num_of_samples = X_test.shape[0]

    for i in range(num_of_samples):
        image = X_test[i]
        test = normalize(image.reshape(1, 28, 28))
        pred = model.predict(test).argmax()
        pred_img = Image.open(digits_path + str(pred) + '.png')

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
    print(test_acc)


input_path = './data/test/'
digits_path = './data/digits/'

model = load_model('network.h5')
X_test, y_test = import_images_from_dir(input_path, model='sequential')
evaluate(X_test, num_of_samples=3)
# evaluate_all(X_test, y_test)
