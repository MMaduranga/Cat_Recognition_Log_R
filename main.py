import numpy as np
import matplotlib.pyplot as plt
import h5py
import copy


train_dataset = h5py.File("data\\train_catvnoncat.h5", "r")  # read data set and set train and test data
x_train_orig = np.array(train_dataset["train_set_x"][:])
y_train_orig = np.array(train_dataset["train_set_y"][:]).reshape(1, x_train_orig.shape[0])
test_dataset = h5py.File("data\\test_catvnoncat.h5", "r")
x_test_orig = np.array(test_dataset["test_set_x"][:])
y_test_orig = np.array(test_dataset["test_set_y"][:]).reshape(1, x_test_orig.shape[0])
classes = np.array(test_dataset["list_classes"][:])


def show_image(index):
    plt.imshow(x_train_orig[:, index])
    # print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")


x_train_reshape = x_train_orig .reshape(x_train_orig .shape[0], -1).T  # reshape test and train data
x_test_reshape = x_test_orig .reshape(x_test_orig.shape[0], -1).T

x_train = x_train_reshape / 255.0  # standardize data set
x_test = x_test_reshape / 255.0


def sigmoid(function):  # sigmoid activation function
    sig_val = 1 / (1 + np.exp(-function))
    return sig_val


def initialize_with_zeros(dim):  # initialize weight and intercept
    weight = np.zeros((dim, 1))
    intercept = 0.0
    return weight, intercept


def propagate(weight, intercept, x, y):
    m = x.shape[1]
    a = sigmoid(np.dot(weight.T, x) + intercept)  # f(x)=A 1xm matrix with calculated sigmoid values for whole data set
    cost = (-1 / m) * np.sum(((y * np.log(a)) + ((1 - y) * np.log(1 - a))))  # cost value
    dw = (1 / m) * (np.dot(x, (a - y).T))  # dJ/dw
    db = (1 / m) * np.sum(a - y)  # dJ/dw J-cost function
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw, "db": db}
    return grads, cost


def optimize(weight_org, intercept_org, x, y, num_iterations=100, learning_rate=0.009):
    weight = copy.deepcopy(weight_org)
    intercept = copy.deepcopy(intercept_org)
    dw, db = 0, 0
    for i in range(num_iterations):
        grads, cost = propagate(weight, intercept, x, y)
        dw = grads["dw"]
        db = grads["db"]
        weight = weight - learning_rate * dw
        intercept = intercept - learning_rate * db
        if i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, float(cost)))  # Print the cost every 100 training iterations
    params = {"w": weight, "b": intercept}
    grads = {"dw": dw, "db": db}
    return params, grads


def predict(weight, intercept, x):
    m = x.shape[1]
    y_prediction = np.zeros((1, m))
    weight = weight.reshape(x.shape[0], 1)
    a = sigmoid(np.dot(weight.T, x) + intercept)
    for i in range(a.shape[1]):
        if a[0, i] > 0.5:
            y_prediction[0, i] = 1
        else:
            y_prediction[0, i] = 0
    return y_prediction


def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5):
    weight, intercept = initialize_with_zeros(x_train.shape[0])
    params, grads = optimize(weight, intercept, x_train, y_train, num_iterations, learning_rate)
    weight = params["w"]
    intercept = params["b"]
    y_prediction_test = predict(weight, intercept, x_test)
    y_prediction_train = predict(weight, intercept, x_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    d = {"y_prediction_test": y_prediction_test, "y_prediction_train": y_prediction_train, "w": weight,
         "b": intercept, "learning_rate": learning_rate, "num_iterations": num_iterations}
    return d


logistic_regression_model = model(x_train, y_train_orig, x_test, y_test_orig, num_iterations=20000, learning_rate=0.005)
for key, val in logistic_regression_model.items():
    print(key, val)
