import pickle

from constants import LR_MODEL_PATH, TEST_PATH
from tools import normalize, import_images_from_dir

# Loading the model from a file
with open(LR_MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Loading the hand-drawn digits
X_test, y_test = import_images_from_dir(TEST_PATH, model=LR_MODEL_PATH)

# Nomalizing the dataset
X_test = normalize(X_test)

test_acc = model.score(X_test, y_test)
print('Accuracy is ' + str(test_acc))

