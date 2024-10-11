import numpy as np


# Linear regression model class
class LinearRegressionModel:
    def __init__(self):
        self.m = 0  # Slope (weight)
        self.b = 0  # Intercept (bias)

    def predict(self, X):
        # Predicted value y = mx + b
        return self.m * X + self.b

    def compute_cost(self, X, y):
        # Mean Squared Error (MSE) cost function
        n = len(y)
        predictions = self.predict(X)
        return (1 / n) * np.sum((predictions - y) ** 2)

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        # Gradient descent algorithm to minimize the cost function
        n = len(y)

        for _ in range(epochs):
            predictions = self.predict(X)
            # Calculate gradients
            dm = -(2 / n) * np.sum(X * (y - predictions))
            db = -(2 / n) * np.sum(y - predictions)

            # Update weights and bias using gradients
            self.m -= learning_rate * dm
            self.b -= learning_rate * db

        return self.m, self.b


# Example dataset
X = np.array([1, 2, 3, 4, 5])  # Features
y = np.array([5, 7, 9, 11, 13])  # Labels (target)

# Instantiate and train the model
model = LinearRegressionModel()
initial_cost = model.compute_cost(X, y)
print(f"Initial cost (before training): {initial_cost}")

# Train the model
model.fit(X, y, learning_rate=0.01, epochs=1000)

# Make predictions
predictions = model.predict(X)
final_cost = model.compute_cost(X, y)
print(f"Final cost (after training): {final_cost}")

# Display the learned slope and intercept
print(f"Slope (m): {model.m}")
print(f"Intercept (b): {model.b}")
print(f"Predictions: {predictions}")