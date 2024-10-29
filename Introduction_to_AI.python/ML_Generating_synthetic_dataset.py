import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

# Generate a synthetic dataset
X = 2 * np.random.rand(100, 1)  # 100 data points
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3X + noise

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred  = model.predict(X_test)

print(f"x: {X}, y: {y}")
# Plot the datta points and the regression line
plt.scatter(X_test, y_test, color="blue", label="Actual data")
plt.plot(X_test, y_pred, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# X = 2 * np.random.rand(100, 1)

# np.random.rand(100, 1): This part generates 100 random numbers between 0 and 1, arranged as a 100x1 column vector. Each number represents one data point on the X-axis.
# 2 * np.random.rand(100, 1): Multiplying by 2 expands the range of each random number from [0, 1] to [0, 2]. So, each X value is a random float between 0 and 2.

# y = 4 + 3 * X + np.random.randn(100, 1)

# 4 + 3 * X: This is the linear equation for y. It means that y depends on X such that:
#
# The intercept (where X = 0) is 4.
# The slope (the rate at which y changes with respect to X) is 3, so for every 1-unit increase in X, y increases by about 3 units.
# np.random.randn(100, 1): This term adds "noise" to the y values. np.random.randn generates 100 random values from a normal distribution with a mean of 0 and a standard deviation of 1. Adding this noise simulates real-world data, where observations don’t fit perfectly on a line but rather fluctuate around it due to various factors.