# use polynomial regression to capture the non-linear relationships.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])  # Two features (e.g., X1 and X2)
Y = np.array([5, 7, 9, 11, 13])  # Dependent variable (target)

# Generate polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit the polynomial model
model = LinearRegression()
model.fit(X_poly, Y)

# Predict values
Y_pred = model.predict(X_poly)

# Display the coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")



