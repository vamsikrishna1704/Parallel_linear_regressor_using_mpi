import numpy as np
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

def find_coefficients(X, y):
    #add a column of ones to X for the intercept term as biases
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    #compute the matrix (X^T X) and the vector (X^T y)
    X_transpose = X.T
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_y = np.dot(X_transpose, y)
   
    coefficients = np.dot(np.linalg.inv(X_transpose_X), X_transpose_y)
    
    return coefficients

def calculate_mse(y_true, y_pred):
    # Calculate the Mean Squared Error between actual and predicted values
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def predict_values(X, coefficients):
    #a a column of ones to X for the intercept term
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    #dot product of X and coefficients is our prediction vector
    predictions = np.dot(X, coefficients)
    return predictions


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

start = time.time()

#load the dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute the coefficients using the training data
coefficients = find_coefficients(X_train, y_train)

# Predict using the test data
test_predictions = predict_values(X_test, coefficients)
print("Linear approach: ")
print("Coeffiecients: ")
print(coefficients)
# Calculate and print the MSE for the test predictions
test_mse = calculate_mse(y_test, test_predictions)
print("Mean Squared Error on Test Data:", test_mse)

end = time.time()
print("total time: "+str(end-start))


import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Plot the predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions, alpha=0.3)
plt.plot([0, 5], [0, 5], '--', color='red')  # Diagonal line representing perfect predictions
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values Linear Approach')
plt.show()