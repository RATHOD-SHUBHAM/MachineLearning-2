import numpy as np

class LinearRegression:

    def __init__(self, lr = 0.001, n_iter = 1000):
        self.lr = lr # learning rate
        self.n_iter = n_iter  # no of iteration
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape # Row and Column
        
        # Initially we set weight and bias to 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            # y = wX + b
            y_pred = np.dot(X, self.weights) + self.bias

            # Optimizer
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update the weight and bias
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
        


    def predict(self, X):
        # y = wX + b
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred