# In multiple linear regression, we extend the concept to multiple features. The equation becomes:
# y = b + b1x1 + b2x2 + ....

# y is the dependent variable,
# x1, x2, ... are the independent variables (features),
# b, b1, b2 .... are the coefficients

import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data (multiple features)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])  # Two features (e.g., X1 and X2)
Y = np.array([5, 7, 9, 11, 13])  # Dependent variable (target)

# Create and fit the model
model = LinearRegression()
model.fit(X, Y)

# Predict values
Y_pred = model.predict(X)

# Display the coefficients and intercept
print(f"Coefficients (b1, b2): {model.coef_}")
print(f"Intercept (b0): {model.intercept_}")

# This will predict Y based on two independent variables (features). The model will compute multiple coefficients for the input features.
