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

# Print the fitted parameters
print(f"Refined fitted parameters:\na1 = {a1}\nb1 = {b1}\na2 = {a2}\nb2 = {b2}\nc = {c}")

# Generate a range of time values for plotting the fitted curve
time_fit_scaled = np.linspace(min(time_us_scaled), max(time_us_scaled), 1000)
voltage_fit = double_exponential_decay(time_fit_scaled, *params)

# Plotting the original data and the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(time_us_scaled, voltage, color='blue', label="Observed Data (Scaled Time)")
plt.plot(time_fit_scaled, voltage_fit, color='red', label=f"Fitted Double Exponential Model")
plt.xlabel("Time (seconds)")
plt.ylabel("Battery Voltage (V)")
plt.title("Battery Voltage vs. Time with Double Exponential Decay Fit (Scaled Time)")
plt.legend()
plt.grid(True)
plt.show()
