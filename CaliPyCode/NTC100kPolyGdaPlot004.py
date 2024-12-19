import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def polynomial_features(x, degree):
    x_poly = np.ones((x.shape[0], 1))
    for i in range(1, degree + 1):
        x_poly = np.hstack((x_poly, x ** i))
    return x_poly

def ridge_regression(X, y, alpha):
    n, d = X.shape
    I = np.eye(d)
    beta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return beta

def predict(X, beta):
    return X @ beta

def mean_squared_error_manual(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def train_test_split_manual(x, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(test_size * x.shape[0])
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]

def update_model():
    degree = int(degree_slider.get())
    alpha = float(alpha_slider.get())
    
    x_poly = polynomial_features(x_scaled, degree)
    X_train, X_test, y_train, y_test = train_test_split_manual(x_poly, y_desired, test_size=0.2, random_state=42)
    beta = ridge_regression(X_train, y_train, alpha)
    y_pred_train = predict(X_train, beta)
    y_pred_test = predict(X_test, beta)
    train_error = mean_squared_error_manual(y_train, y_pred_train)
    test_error = mean_squared_error_manual(y_test, y_pred_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_desired, color='blue', label='Actual')
    plt.scatter(x, predict(x_poly, beta), color='red', label='Predicted')
    plt.xlabel('Frequency')
    plt.ylabel('Temperature')
    plt.legend()
    plt.title(f'Train Error: {train_error:.4f}, Test Error: {test_error:.4f}')
    plt.show()

# Load Data
file_path_old = r'D:\_July2024\AparProj-2\Code30072024\CaliCode01\Dt1M_555_Col3small.csv'
data = pd.read_csv(file_path_old, header=None)

x = data.iloc[:, 2].values.reshape(-1, 1)  # Frequency (Column 2)
y_desired = data.iloc[:, 1].values  # Temperature (Column 1)

# Normalize features manually
x_mean = np.mean(x)
x_std = np.std(x)
x_scaled = (x - x_mean) / x_std

# Create the main window
root = tk.Tk()
root.title("Polynomial Regression Tuning")

# Create and place sliders
degree_label = tk.Label(root, text="Polynomial Degree")
degree_label.pack()
degree_slider = tk.Scale(root, from_=1, to_=10, orient=tk.HORIZONTAL)
degree_slider.set(5)
degree_slider.pack()

alpha_label = tk.Label(root, text="Alpha (Regularization Strength)")
alpha_label.pack()
alpha_slider = tk.Scale(root, from_=1e-6, to_=1e1, resolution=1e-6, orient=tk.HORIZONTAL)
alpha_slider.set(1.0)
alpha_slider.pack()

# Create and place the update button
update_button = tk.Button(root, text="Update Model", command=update_model)
update_button.pack()

# Run the application
root.mainloop()
