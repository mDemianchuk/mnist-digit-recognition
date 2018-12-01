import pickle
from operator import itemgetter

from joblib import Parallel, delayed
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def grid_search(X_train, X_test, y_train, y_test, c_pow):
    best_params = [0, None, None, None]
    for penalty in ['l1', 'l2']:
        # Not every solver is compatible with some penalty types
        if penalty == 'l1':
            for solver in ['liblinear', 'saga']:
                best_params = \
                    evaluate(X_train, X_test, y_train, y_test, 10 ** c_pow, penalty, solver, best_params)
        else:
            for solver in ['newton-cg', 'lbfgs', 'sag']:
                best_params = \
                    evaluate(X_train, X_test, y_train, y_test, 10 ** c_pow, penalty, solver, best_params)

    return best_params


def evaluate(X_train, X_test, y_train, y_test, C, penalty, solver, best_params):
    model = LogisticRegression(C=C, penalty=penalty, solver=solver)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    if acc > best_params[0]:
        best_params = [acc, C, penalty, solver]
    return best_params


# Loading the MNIST dataset
mnist_df = fetch_mldata('MNIST original')

X_train, X_test, y_train, y_test = train_test_split(
    mnist_df.data, mnist_df.target, test_size=1 / 7.0, random_state=0)

# Due to time constraints we use only 1/20 of the dataset
# Comment out lines 43-49 to use the entire dataset 
# Picking the first 3000 samples as the train set
X_train = X_train[:3000, :]
y_train = y_train[:3000]

# Picking the first 500 samples as the test set
X_test = X_test[:500, :]
y_test = y_test[:500]

# Running the grid search in parallel
result = Parallel(n_jobs=4)(delayed(grid_search)(X_train, X_test, y_train, y_test, c_pow) for c_pow in range(-3, 3))
result = sorted(result, key=itemgetter(0), reverse=True)
best_c = result[0][1]
best_penalty = result[0][2]
best_solver = result[0][3]

# Creating the model with the optimum parameters
model = LogisticRegression(C=best_c, penalty=best_penalty, solver=best_solver)
model.fit(X_train, y_train)

# Saving the model into a HDF5 file
model_filename = 'logistic_regression.h5'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
