from mpi4py import MPI
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import time


def calculate_mse(y_true, y_pred):
    # Calculate the Mean Squared Error between actual and predicted values
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def predict_values(X, coefficients):
    # Add a column of ones to X for the intercept term
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Calculate predictions: dot product of X and coefficients
    predictions = np.dot(X, coefficients)
    return predictions

def parallel_find_coefficients(X, y):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #scatter data to all processes
    if len(X) is None:
        return
    num_samples_per_process = len(X) // size
    X_local = np.zeros((num_samples_per_process, X.shape[1]))
    y_local = np.zeros(num_samples_per_process)

    comm.Scatter(X, X_local, root=0)
    comm.Scatter(y, y_local, root=0)

    #add a column of ones to X_local for the intercept term
    X_local = np.hstack([np.ones((X_local.shape[0], 1)), X_local])

    #local computations of (X^T * X) and (X^T * y)
    X_transpose_X_local = np.dot(X_local.T, X_local)
    X_transpose_y_local = np.dot(X_local.T, y_local)

    #reduce all local computations to a single result on the root process
    X_transpose_X_global = np.zeros_like(X_transpose_X_local)
    X_transpose_y_global = np.zeros_like(X_transpose_y_local)

    comm.Reduce(X_transpose_X_local, X_transpose_X_global, op=MPI.SUM, root=0)
    comm.Reduce(X_transpose_y_local, X_transpose_y_global, op=MPI.SUM, root=0)

    if rank == 0:
        #only root computes the final coefficients
        coefficients = np.dot(np.linalg.inv(X_transpose_X_global), X_transpose_y_global)
        return coefficients

def main():
    print("Parallel Approach")
    if MPI.COMM_WORLD.Get_rank() == 0:
        # Load data
        data = fetch_california_housing()
        X = data.data
        y = data.target

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train = None
        y_train = None

    start = time.time()
    # Compute coefficients in parallel
    coefficients = parallel_find_coefficients(X_train, y_train)
    end = time.time()
    if MPI.COMM_WORLD.Get_rank() == 0:
        # Continue with the main process tasks
        print("Coefficients:", coefficients)

    test_predictions = predict_values(X_test, coefficients)

# Calculate and print the MSE for the test predictions
    test_mse = calculate_mse(y_test, test_predictions)
    print("Mean Squared Error on Test Data:", test_mse)

    print("time taken: "+str(end-start))

    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt

    # Plot the predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_predictions, alpha=0.3)
    plt.plot([0, 5], [0, 5], '--', color='red')  # Diagonal line representing perfect predictions
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values Parallel Approach')
    plt.show()

if __name__ == '__main__':
    main()


# Predict using the test data
