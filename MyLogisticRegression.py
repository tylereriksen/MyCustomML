import numpy as np

class MyLogisticRegression:

    def __init__(self, activation):
        self.weights = None
        self.activation = activation

    def fit(self, X, y, lr, iterations):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        self.weights = np.random.rand(X.shape[1] + 1)

        for _ in range(iterations):
            y_pred = self.activation(X_aug @ self.weights)
            gradient = - (1 * X_aug.T @ (y - y_pred)) / X.shape[0]
            self.weights = self.weights - lr * gradient


    # use the model to predict the outcome
    def predict(self, X: np.ndarray, threshold=0.5) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Fit the model before prediction")
        
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return (self.activation(X_aug @ self.weights) >= threshold).astype(int)

    
    # get the weights of the model
    def get_weights(self):
        return self.weights

    
    # get the binary cross-entropy loss from the ouput and predicted outputs
    def get_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
