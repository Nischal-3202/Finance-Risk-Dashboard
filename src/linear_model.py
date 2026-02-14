import numpy as np


class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, num_iterations=2000, lambda_=0.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.w = None
        self.b = None
        self.cost_history = []

    def compute_cost(self, X, y):
        m = X.shape[0]

        predictions = np.dot(X, self.w) + self.b
        error = predictions - y

        cost = (1/(2*m)) * np.sum(error**2)

        reg_cost = (self.lambda_ / (2*m)) * np.sum(self.w**2)

        return cost + reg_cost

    def compute_gradient(self, X, y):
        m = X.shape[0]

        predictions = np.dot(X, self.w) + self.b
        error = predictions - y

        dj_dw = (1/m) * np.dot(X.T, error)
        dj_db = (1/m) * np.sum(error)

        dj_dw += (self.lambda_ / m) * self.w

        return dj_dw, dj_db

    def fit(self, X, y):
        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.num_iterations):
            dj_dw, dj_db = self.compute_gradient(X, y)

            self.w -= self.learning_rate * dj_dw
            self.b -= self.learning_rate * dj_db

            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

            if i % 200 == 0:
                print(f"Iteration {i}, Cost: {cost:.4f}")

    def predict(self, X):
        return np.dot(X, self.w) + self.b