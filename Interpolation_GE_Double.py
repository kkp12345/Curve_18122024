import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Provided data
voltage = np.array([4.48, 4.27, 4.00, 3.75, 3.48, 3.23, 3.00, 2.75, 2.50, 2.25, 2.00])
time_us = np.array([44000, 47000, 52000, 57000, 64000, 70000, 81000, 94000, 110000, 140000, 200000])

# Scale down time to prevent overflow
time_us_scaled = time_us / 1e6  # Convert to seconds

# Define the double exponential decay model
def double_exponential_decay(x, a1, b1, a2, b2, c):
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x) + c

# Set refined bounds for each parameter
lower_bounds = [0, -50, 0, -50, 0]  # Allow decay rates to be more flexible
upper_bounds = [10, 0, 10, 0, 5]    # Upper bounds for amplitudes and baseline

# Provide initial guesses close to expected values
initial_guesses = [5, -1, 1, -0.1, 2]  # Initial guesses for a1, b1, a2, b2, c

# Fit the model with adjusted bounds and initial guesses
params, _ = curve_fit(double_exponential_decay, time_us_scaled, voltage, p0=initial_guesses, bounds=(lower_bounds, upper_bounds))
a1, b1, a2, b2, c = params

# Generate predicted voltages using the fitted model
voltage_predicted_double = double_exponential_decay(time_us_scaled, *params)

# Interpolation model
interpolator = interp1d(time_us_scaled, voltage, kind='cubic')
voltage_predicted_interp = interpolator(time_us_scaled)

# Calculate MAPE and R-squared for Double Exponential Decay Model
mape_double = np.mean(np.abs((voltage - voltage_predicted_double) / voltage)) * 100
ss_total_double = np.sum((voltage - np.mean(voltage))**2)
ss_residual_double = np.sum((voltage - voltage_predicted_double)**2)
r_squared_double = 1 - (ss_residual_double / ss_total_double)

# Calculate MAPE and R-squared for Interpolation Model
mape_interp = np.mean(np.abs((voltage - voltage_predicted_interp) / voltage)) * 100
ss_total_interp = np.sum((voltage - np.mean(voltage))**2)
ss_residual_interp = np.sum((voltage - voltage_predicted_interp)**2)
r_squared_interp = 1 - (ss_residual_interp / ss_total_interp)

# Print the fitted parameters and accuracy metrics
print(f"Refined fitted parameters for Double Exponential Decay Model:\n"
      f"a1 = {a1:.4f}, b1 = {b1:.4f}, a2 = {a2:.4f}, b2 = {b2:.4f}, c = {c:.4f}")
print(f"\nAccuracy Metrics for Double Exponential Decay Model:\n"
      f"Mean Absolute Percentage Error (MAPE): {mape_double:.2f}%\n"
      f"Coefficient of Determination (R²): {r_squared_double:.4f}")

print(f"\nAccuracy Metrics for Interpolation Model:\n"
      f"Mean Absolute Percentage Error (MAPE): {mape_interp:.2f}%\n"
      f"Coefficient of Determination (R²): {r_squared_interp:.4f}")

# Generate a range of time values for plotting the fitted curve
time_fit_scaled = np.linspace(min(time_us_scaled), max(time_us_scaled), 1000)
voltage_fit_double = double_exponential_decay(time_fit_scaled, *params)
voltage_fit_interp = interpolator(time_fit_scaled)

# Plotting the original data and the fitted curves
plt.figure(figsize=(10, 6))
plt.scatter(time_us_scaled, voltage, color='blue', label="Observed Data (Scaled Time)")
plt.plot(time_fit_scaled, voltage_fit_double, color='red', label=f"Double Exponential Model (MAPE = {mape_double:.2f}%, R² = {r_squared_double:.4f})")
plt.plot(time_fit_scaled, voltage_fit_interp, color='green', linestyle='--', label=f"Interpolation Model (MAPE = {mape_interp:.2f}%, R² = {r_squared_interp:.4f})")
plt.xlabel("Time (seconds)")
plt.ylabel("Battery Voltage (V)")
plt.title("Battery Voltage vs. Time with Double Exponential and Interpolation Models")
plt.legend()
plt.grid(True)
plt.show()
