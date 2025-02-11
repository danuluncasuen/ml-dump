import numpy as np
import matplotlib.pyplot as plt

def generate_random_data():
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1) #adding some noise
    return X, y

def plot(e):
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


class LinearRegression:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.theta_0 = 0  # Intercept
        self.theta_1 = 0  # Slope

    def fit(self, X, y):
        m = len(X)
        
        for _ in range(self.n_iters):
            y_predicted = self.theta_0 + self.theta_1 * X  # Hypothesis
            
            # Compute gradients
            d_theta_0 = -(2/m) * np.sum(y - y_predicted)
            d_theta_1 = -(2/m) * np.sum((y - y_predicted) * X)
            
            # Update parameters
            self.theta_0 -= self.learning_rate * d_theta_0
            self.theta_1 -= self.learning_rate * d_theta_1

    def predict(self, X):
        return self.theta_0 + self.theta_1 * X


if __name__ == '__main__':
    model = LinearRegression(learning_rate=0.1, n_iters=1000)
    X,y = generate_random_data()
    plt.scatter(X, y, label="Randomly Generated Records")
    model.fit(X, y)
    # Make predictions
    y_prediction = model.predict(X)
    plt.scatter(X, y, label="Generated Records")
    plt.plot(X, y_prediction, color="red", linewidth=2, label="Regression line")
    plot(plt)