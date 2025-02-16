import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0  # Step function

    def predict(self, X):
        return self.activation(np.dot(X, self.weights) + self.bias)

    def train(self, X_train, y_train):
        for _ in range(self.epochs):
            for X, y in zip(X_train, y_train):
                prediction = self.predict(X)
                error = y - prediction

                # Update weights and bias
                self.weights += self.learning_rate * error * X
                self.bias += self.learning_rate * error


if __name__ == '__main__':
    # Example: Training a perceptron to learn the AND function
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])  # AND logic gate

    perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
    perceptron.train(X_train, y_train)

    # Testing
    for X in X_train:
        print(f"Input: {X}, Prediction: {perceptron.predict(X)}")