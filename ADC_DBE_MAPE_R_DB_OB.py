import numpy as np
from scipy.optimize import curve_fit
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
voltage_predicted = double_exponential_decay(time_us_scaled, *params)

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((voltage - voltage_predicted) / voltage)) * 100

# Calculate R² (Coefficient of Determination)
ss_total = np.sum((voltage - np.mean(voltage))**2)
ss_residual = np.sum((voltage - voltage_predicted)**2)
r_squared = 1 - (ss_residual / ss_total)

# Print the fitted parameters and accuracy metrics
print(f"Refined fitted parameters:\n"
      f"a1 = {a1:.4f}, b1 = {b1:.4f}, a2 = {a2:.4f}, b2 = {b2:.4f}, c = {c:.4f}")
print(f"\nAccuracy Metrics:\n"
      f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n"
      f"Coefficient of Determination (R²): {r_squared:.4f}")

# Generate a range of time values for plotting the fitted curve
time_fit_scaled = np.linspace(min(time_us_scaled), max(time_us_scaled), 1000)
voltage_fit = double_exponential_decay(time_fit_scaled, *params)

# Plotting the original data and the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(time_us_scaled, voltage, color='blue', label="Observed Data (Scaled Time)")
plt.plot(time_fit_scaled, voltage_fit, color='red', label=f"Fitted Double Exponential Model")
plt.xlabel("Time (seconds)")
plt.ylabel("Battery Voltage (V)")

# Update the title to include MAPE and R²
plt.title(f"Battery Voltage vs. Time with Double Exponential Decay Fit\n"
          f"MAPE = {mape:.2f}% | R² = {r_squared:.4f}")

plt.legend()
plt.grid(True)
plt.show()
