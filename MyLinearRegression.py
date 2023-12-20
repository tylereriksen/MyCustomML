import numpy as np
import matplotlib.pyplot as plt
from multipledispatch import dispatch

# class to implement my custom Linear Regression
class MyLinearRegression:

    # initialize the weights
    def __init__(self):
        self.weights = None


    # fit the model on the input and output data using the equation form
    # typically used for smaller datasets or data with a small number of features
    # no hyperparameter tuning involved
    @dispatch(np.ndarray, np.ndarray)
    def fit(self, X, y):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        self.weights = np.linalg.pinv(X_aug.T @ X_aug) @ X_aug.T @ y


    # fit the model on the input and output data using gradient descent
    # typically used for larger datasets or large number of features and has objective function flexibility
    # has better scalability comapratively to the equation form
    @dispatch(np.ndarray, np.ndarray, float, int)
    def fit(self, X, y, lr, iterations):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        self.weights = np.zeros(X.shape[1] + 1)
        losses = []

        for _ in range(iterations):
            y_pred = X_aug @ self.weights
            losses.append(self.get_loss(y, y_pred))
            gradients = -2 / X.shape[0] * X_aug.T @ (y - y_pred)
            self.weights = self.weights - lr * gradients

        losses.append(self.get_loss(y, X_aug @ self.weights))
        plt.plot(range(len(losses)), losses, 'k')
        plt.grid()
        plt.show()

    
    # TODO
    def fit_lasso(self, X, y, alpha, iterations):
        pass


    @dispatch(np.ndarray, np.ndarray, float)
    def fit_ridge(self, X, y, alpha):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        I = np.eye(X_aug.shape[1])
        I[-1, -1] = 0
        self.weights = np.linalg.inv(X_aug.T @ X_aug + alpha * I) @ X_aug.T @ y

    
    @dispatch(np.ndarray, np.ndarray, float, float, int)
    def fit_ridge(self, X, y, alpha, lr, iterations):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        self.weights = np.zeros(X_aug.shape[1])
        for _ in range(iterations):
            y_pred = X_aug @ self.weights
            gradient = -2 / X.shape[0] * X_aug.T @ (y - y_pred) + 2 * alpha * np.append(self.weights[:-1], 0)
            self.weights = self.weights - lr * gradient


    # use the model to predict the outcome
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Fit the model before prediction")
        
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_aug @ self.weights
    

    # get the weights of the model
    def get_weights(self):
        return self.weights
    

    # get the mse from the output and predicted outputs
    def get_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y - y_pred) ** 2)
