import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

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

def percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_test_split_manual(x, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    test_set_size = int(test_size * x.shape[0])
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]

def load_data():
    global x, y_desired, x_scaled, file_path
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path, header=None)
        x = data.iloc[:, 2].values.reshape(-1, 1)  # Frequency (Column 2)
        y_desired = data.iloc[:, 1].values  # Temperature (Column 1)

        # Normalize features manually
        x_mean = np.mean(x)
        x_std = np.std(x)
        x_scaled = (x - x_mean) / x_std

        messagebox.showinfo("Data Loaded", f"Data loaded from {file_path}")
    else:
        messagebox.showwarning("No File Selected", "Please select a valid data file.")

def save_results(beta, x_poly):
    if x is None or y_desired is None:
        messagebox.showwarning("No Data", "Please load data before saving results.")
        return

    y_mean = np.mean(y_desired)
    y_std = np.std(y_desired)
    y_scaled = (y_desired - y_mean) / y_std

    output_data = pd.DataFrame({
        'Frequency': x.flatten(),
        'Actual Temperature': y_desired,
        'Normalized Frequency': x_scaled.flatten(),
        'Normalized Temperature': y_scaled.flatten(),
        'Predicted Temperature': predict(x_poly, beta)
    })
    output_data['Error'] = output_data['Actual Temperature'] - output_data['Predicted Temperature']

    output_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if output_file_path:
        output_data.to_csv(output_file_path, index=False)
        messagebox.showinfo("Results Saved", f"Results saved to {output_file_path}")
    else:
        messagebox.showwarning("No File Selected", "Please select a valid file path to save the results.")

def update_model():
    if x is None or y_desired is None:
        messagebox.showwarning("No Data", "Please load data before updating the model.")
        return

    degree = int(degree_slider.get())
    alpha = float(alpha_slider.get())
    
    x_poly = polynomial_features(x_scaled, degree)
    X_train, X_test, y_train, y_test = train_test_split_manual(x_poly, y_desired, test_size=0.2, random_state=42)
    beta = ridge_regression(X_train, y_train, alpha)
    y_pred_train = predict(X_train, beta)
    y_pred_test = predict(X_test, beta)

    train_error = percentage_error(y_train, y_pred_train)
    test_error = percentage_error(y_test, y_pred_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y_desired, color='blue', label='Actual')
    plt.scatter(x, predict(x_poly, beta), color='red', label='Predicted')
    plt.xlabel('Frequency')
    plt.ylabel('Temperature')
    plt.legend()

    # Update the title to show percentage errors and file name
    file_name = file_path.split('/')[-1] if file_path else 'Unknown File'
    plt.title(f'{file_name} - Train Error: {train_error:.2f}%, Test Error: {test_error:.2f}%')
    
    plt.show()

    save_results(beta, x_poly)

# Initialize global variables
x = None
y_desired = None
x_scaled = None
file_path = None

# Create the main window
root = tk.Tk()
root.title("Polynomial Regression Tuning")

# Create and place the "Load Data" button
load_button = tk.Button(root, text="Load Data", command=load_data)
load_button.pack()

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

# Create and place the "Update Model" button
update_button = tk.Button(root, text="Update Model", command=update_model)
update_button.pack()

# Run the application
root.mainloop()
