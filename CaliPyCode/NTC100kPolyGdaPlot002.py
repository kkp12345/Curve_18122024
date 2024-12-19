import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def get_filename(previousPath):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    initial_dir = os.path.dirname(previousPath)  # Get the directory part of the path
    initial_file = os.path.basename(previousPath)  # Get the file name part of the path
    file_path = filedialog.askopenfilename(initialdir=initial_dir, initialfile=initial_file)
    return file_path

def polynomial_features(x, degree):
    x_poly = np.ones((x.shape[0], 1))
    for i in range(1, degree + 1):
        x_poly = np.hstack((x_poly, x ** i))
    return x_poly

def train_test_split_manual(x, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(test_size * x.shape[0])
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]

def ridge_regression(X, y, alpha):
    # Closed-form solution to Ridge Regression
    n, d = X.shape
    I = np.eye(d)
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return beta

def predict(X, beta):
    return X @ beta

def mean_squared_error_manual(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Load Data
file_path_old = r'D:\_July2024\AparProj-2\Code30072024\CaliCode01\Dt1M_555_Col3small.csv'
file_path = get_filename(file_path_old)
if file_path == '':
    file_path = file_path_old
print("Selected file:", file_path)
data = pd.read_csv(file_path, header=None)

x = data.iloc[:, 2].values.reshape(-1, 1)  # Frequency (Column 2)
y_desired = data.iloc[:, 1].values  # Temperature (Column 1)

# Normalize features manually
x_mean = np.mean(x)
x_std = np.std(x)
x_scaled = (x - x_mean) / x_std

# Polynomial Features
degree = 5  # Increased polynomial degree
x_poly = polynomial_features(x_scaled, degree)

# Split the data manually
X_train, X_test, y_train, y_test = train_test_split_manual(x_poly, y_desired, test_size=0.2, random_state=42)

# Model - Ridge Regression with manual implementation
alpha_values = np.logspace(-6, 6, 13)  # Different values of alpha for cross-validation
best_alpha = alpha_values[0]
best_error = float('inf')

for alpha in alpha_values:
    beta = ridge_regression(X_train, y_train, alpha)
    y_pred_val = predict(X_test, beta)
    error = mean_squared_error_manual(y_test, y_pred_val)
    if error < best_error:
        best_error = error
        best_alpha = alpha

print(f'Best alpha: {best_alpha}')
beta = ridge_regression(X_train, y_train, best_alpha)

# Predictions
y_pred_train = predict(X_train, beta)
y_pred_test = predict(X_test, beta)

# Evaluation
train_error = mean_squared_error_manual(y_train, y_pred_train)
test_error = mean_squared_error_manual(y_test, y_pred_test)
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

# Extract and print the coefficients
intercept = beta[0]
coef = beta[1:]

# Printing the intercept and coefficients for polynomial terms
print(f'Intercept: {intercept}')
for i, c in enumerate(coef):
    print(f'Coefficient for x^{i+1}: {c}')

# Plotting
plt.scatter(x, y_desired, color='blue', label='Actual')
plt.scatter(x, predict(x_poly, beta), color='red', label='Predicted')
plt.xlabel('Frequency')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Normalize Temperature manually
y_mean = np.mean(y_desired)
y_std = np.std(y_desired)
y_scaled = (y_desired - y_mean) / y_std

# Prepare data for saving
output_data = pd.DataFrame({
    'Frequency': x.flatten(),
    'Actual Temperature': y_desired,
    'Normalized Frequency': x_scaled.flatten(),
    'Normalized Temperature': y_scaled.flatten(),
    'Predicted Temperature': predict(x_poly, beta)
})
output_data['Error'] = output_data['Actual Temperature'] - output_data['Predicted Temperature']

# Save the DataFrame to a CSV file
output_data.to_csv('D:/_July2024/AparProj-2/Code30072024/CaliCode01/Predicted_Temperature.csv', index=False)

print("Model coefficients and predictions saved to Predicted_Temperature.csv")
