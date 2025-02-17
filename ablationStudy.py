import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def fit_linear_regression(X, y):
    """
    Fit a linear regression model using the normal equation.
    X: Feature matrix (numpy array)
    y: Target vector (numpy array)
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term (x0 = 1)
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta

def predict(X, theta):
    """
    Make predictions using the linear regression model.
    X: Feature matrix (numpy array)
    theta: Coefficients vector (numpy array)
    """
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term
    return X_b @ theta

def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error of the model.
    y_true: Actual values (numpy array)
    y_pred: Predicted values (numpy array)
    """
    return np.mean((y_true - y_pred) ** 2)

def perform_ablation_study(data):
    """
    Perform an ablation study by removing each feature one at a time, refitting the model, and computing metrics.
    """
    original_features = data.drop('price', axis=1)
    y = data['price'].values  # Normalize the price for numerical stability

    # Fit the model with all features
    X = original_features.values
    theta = fit_linear_regression(X, y)
    y_pred = predict(X, theta)
    original_mse = mean_squared_error(y, y_pred)
    original_r2 = r2_score(y, y_pred)

    print('All features:')
    print('Model coefficients:', theta)
    print('Mean Squared Error:', original_mse)
    print("R2 Score: {:.2f}".format(original_r2))

    # Ablation study
    for column in original_features.columns:
        X_ablated = original_features.drop(column, axis=1).values
        theta_ablated = fit_linear_regression(X_ablated, y)
        y_pred_ablated = predict(X_ablated, theta_ablated)
        mse_ablated = mean_squared_error(y, y_pred_ablated)
        r2_ablated = r2_score(y, y_pred_ablated)

        print(f'\nDropped column: {column}')
        print('Model coefficients:', theta_ablated)
        print('Mean Squared Error:', mse_ablated)
        print("R2 Score: {:.2f}".format(r2_ablated))

# Load your data
data = pd.read_csv('dataset/cleanedData.csv')

# Perform the ablation study
perform_ablation_study(data)
