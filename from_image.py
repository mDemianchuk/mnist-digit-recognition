from keras.models import load_model
from tools import import_images_from_dir

model = load_model('network.h5')
# print(model.predict_proba(X_test))
# print(model.predict_classes(X_test))

input = './data/test/'
allimages, alllabels = import_images_from_dir(input)

test_loss, test_acc = model.evaluate(allimages, alllabels)
print(test_acc)
