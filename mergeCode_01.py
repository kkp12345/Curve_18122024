import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ============================
# Generalized Analytical Model
# ============================

def calculate_voltage(t_th, R, C, Vcc):
    """
    Calculate Vin using the analytical formula.
    :param t_th: Time in seconds (array or scalar)
    :param R: Resistance in ohms
    :param C: Capacitance in farads
    :param Vcc: Supply voltage
    :return: Calculated Vin
    """
    return (0.5 * Vcc) / (1 - np.exp(-t_th / (R * C)))

# ============================
# Double Exponential Decay Fit
# ============================

# Define the double exponential decay model
def double_exponential_decay(x, a1, b1, a2, b2, c):
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x) + c

def fit_exponential_decay(time_us, voltage):
    """
    Fit the double exponential decay model and return fitted parameters.
    :param time_us: Observed time in microseconds
    :param voltage: Observed battery voltage
    :return: Fitted parameters and time for plotting the curve
    """
    # Scale down time to seconds
    time_us_scaled = time_us / 1e6

    # Initial guesses and bounds for curve fitting
    initial_guesses = [5, -1, 1, -0.1, 2]
    lower_bounds = [0, -50, 0, -50, 0]
    upper_bounds = [10, 0, 10, 0, 5]

    # Fit the model
    params, _ = curve_fit(double_exponential_decay, time_us_scaled, voltage,
                          p0=initial_guesses, bounds=(lower_bounds, upper_bounds))
    return params, time_us_scaled

# ============================
# Main Plot for Comparison
# ============================

# Data for fitting the exponential model
voltage = np.array([4.48, 4.27, 4.00, 3.75, 3.48, 3.23, 3.00, 2.75, 2.50, 2.25, 2.00])
time_us = np.array([44000, 47000, 52000, 57000, 64000, 70000, 81000, 94000, 110000, 140000, 200000])

# Parameters for the generalized model
parameters = [
    (100000, 0.000001, 3.3),  # Example set 1: R=100kΩ, C=1µF, Vcc=3.3V
]

# Time range for the analytical model
t_th_range = (0.04, 0.2)

# Create a single plot for comparison
plt.figure(figsize=(10, 6))

# Generalized Analytical Model
t_th = np.linspace(t_th_range[0], t_th_range[1], 500)  # Generate time values
for R, C, Vcc in parameters:
    V_in = calculate_voltage(t_th, R, C, Vcc)  # Calculate voltage
    plt.plot(t_th, V_in, linestyle='--', label=f'Generalized: R={R}Ω, C={C}F, Vcc={Vcc}V')

# Double Exponential Decay Fit
params, time_us_scaled = fit_exponential_decay(time_us, voltage)
time_fit_scaled = np.linspace(min(time_us_scaled), max(time_us_scaled), 1000)
voltage_fit = double_exponential_decay(time_fit_scaled, *params)

# Plot observed data and fitted curve
plt.scatter(time_us_scaled, voltage, color='blue', label="Observed Data")
plt.plot(time_fit_scaled, voltage_fit, color='red', label="Fitted Double Exp. Model")

# Plot settings
# plt.title("Comparison of Generalized Analytical Model and Double Exponential Decay Fit with Observed Data")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.grid(True)
plt.legend()
plt.show()

# Print the fitted parameters for the double exponential decay
a1, b1, a2, b2, c = params
print(f"Fitted Parameters:\na1 = {a1:.4f}\nb1 = {b1:.4f}\na2 = {a2:.4f}\nb2 = {b2:.4f}\nc = {c:.4f}")
