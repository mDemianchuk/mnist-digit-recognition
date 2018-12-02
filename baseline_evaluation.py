import pickle

from tools import normalize, import_images_from_dir

# Loading the model from a file
model_filename = 'logistic_regression.h5'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Loading the hand-drawn digits
input_path = './data/test/'
X_test, y_test = import_images_from_dir(input_path, model='lr')

# Nomalizing the dataset
X_test = normalize(X_test)

test_acc = model.score(X_test, y_test)
print('Accuracy is ' + str(test_acc) + '%')
