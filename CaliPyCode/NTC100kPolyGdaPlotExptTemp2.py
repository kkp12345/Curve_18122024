# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

def get_filename(previousPath):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    initial_dir = os.path.dirname(previousPath)  # Get the directory part of the path
    initial_file = os.path.basename(previousPath)  # Get the file name part of the path
    file_path = filedialog.askopenfilename(initialdir=initial_dir, initialfile=initial_file)
    return file_path

# Load Data
file_path_old = r'D:\_July2024\AparProj-2\Code30072024\CaliCode01\Dt1M_555_Col3small.csv'
file_path = get_filename(file_path_old)
if file_path == '':
    file_path = file_path_old
print("Selected file:", file_path)
data = pd.read_csv(file_path, header=None)

x = data.iloc[:, 2].values.reshape(-1, 1)  # Frequency (Column 2)
y_desired = data.iloc[:, 1].values  # Temperature (Column 1)

# Normalize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Polynomial Features
degree = 5  # Increased polynomial degree
poly = PolynomialFeatures(degree=degree)
x_poly = poly.fit_transform(x_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x_poly, y_desired, test_size=0.2, random_state=42)

# Model - Ridge Regression with Cross-Validation
model = Ridge(alpha=1.0)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluation
train_error = mean_squared_error(y_train, y_pred_train)
test_error = mean_squared_error(y_test, y_pred_test)
print(f'Train Error: {train_error}')
print(f'Test Error: {test_error}')

# Extract and print the coefficients
coef = model.coef_
intercept = model.intercept_

# Printing the intercept and coefficients for polynomial terms
print(f'Intercept: {intercept}')
print(f'Coefficients: {coef}')

# Assuming the polynomial features are [1, x, x^2, x^3, x^4, x^5], and the equation is:
# F(x) = P0 + P1*x + P2*x^2 + P3*x^3 + P4*x^4 + P5*x^5
# Intercept is P0, coef[1] is P1, coef[2] is P2, etc.
P0 = intercept
P1 = coef[1]
P2 = coef[2]
P3 = coef[3]
P4 = coef[4]
P5 = coef[5]

# Printing the coefficients for the polynomial terms
print(f'P0: {P0}')
print(f'P1: {P1}')
print(f'P2: {P2}')
print(f'P3: {P3}')
print(f'P4: {P4}')
print(f'P5: {P5}')

# Plotting
plt.scatter(x, y_desired, color='blue', label='Actual')
plt.scatter(x, model.predict(x_poly), color='red', label='Predicted')
plt.xlabel('Frequency')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Save the model coefficients to a CSV file
# output_data = pd.DataFrame({'Frequency': x.flatten(), 'Actual Temperature': y_desired, 'Predicted Temperature': model.predict(x_poly)})
# output_data['Error'] = output_data['Actual Temperature'] - output_data['Predicted Temperature']
# output_data.to_csv('D:/_July2024/AparProj-2/Code29072024/CalibrationCodeNow/Predicted_Temperature.csv', index=False)

# print("Model coefficients and predictions saved to Predicted_Temperature.csv")

# Normalize Temperature
temperature_scaler = StandardScaler()
y_scaled = temperature_scaler.fit_transform(y_desired.reshape(-1, 1))

# Prepare data for saving
output_data = pd.DataFrame({
    'Frequency': x.flatten(),
    'Actual Temperature': y_desired,
    'Normalized Frequency': x_scaled.flatten(),
    'Normalized Temperature': y_scaled.flatten(),
    'Predicted Temperature': model.predict(x_poly)
})
output_data['Error'] = output_data['Actual Temperature'] - output_data['Predicted Temperature']

# Save the DataFrame to a CSV file
output_data.to_csv('D:/_July2024/AparProj-2/Code30072024/CaliCode01/Predicted_Temperature.csv', index=False)

print("Model coefficients and predictions saved to Predicted_Temperature.csv")
