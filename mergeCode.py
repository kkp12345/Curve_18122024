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

def plot_generalized_voltage(ax, parameters, t_th_range):
    """
    Plot Vin as a function of time for different R, C, Vcc values.
    :param ax: Matplotlib axis object for subplotting
    :param parameters: List of tuples [(R, C, Vcc), ...]
    :param t_th_range: Time range for plotting
    """
    t_th = np.linspace(t_th_range[0], t_th_range[1], 500)  # Generate time values
    for R, C, Vcc in parameters:
        V_in = calculate_voltage(t_th, R, C, Vcc)  # Calculate voltage
        ax.plot(t_th, V_in, label=f'Generalized: R={R}Ω, C={C}F, Vcc={Vcc}V')
    ax.set_title("Generalized Analytical Model")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True)
    ax.legend()

# ============================
# Double Exponential Decay Fit
# ============================

# Define the double exponential decay model
def double_exponential_decay(x, a1, b1, a2, b2, c):
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x) + c

def fit_and_plot_exponential_decay(ax, time_us, voltage):
    """
    Fit a double exponential decay model and plot the results.
    :param ax: Matplotlib axis object for subplotting
    :param time_us: Observed time in microseconds
    :param voltage: Observed battery voltage
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
    a1, b1, a2, b2, c = params

    # Generate fitted curve
    time_fit_scaled = np.linspace(min(time_us_scaled), max(time_us_scaled), 1000)
    voltage_fit = double_exponential_decay(time_fit_scaled, *params)

    # Plot observed data and fitted curve
    ax.scatter(time_us_scaled, voltage, color='blue', label="Observed Data")
    ax.plot(time_fit_scaled, voltage_fit, color='red', label="Fitted Double Exp. Model")
    ax.set_title("Double Exponential Decay Fit")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Battery Voltage (V)")
    ax.grid(True)
    ax.legend()

    # Print the fitted parameters
    print(f"Fitted Parameters:\na1 = {a1}\nb1 = {b1}\na2 = {a2}\nb2 = {b2}\nc = {c}")

# ============================
# Main Plot with Subplots
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

# Create subplots to compare both models
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot the generalized analytical model
plot_generalized_voltage(axes[0], parameters, t_th_range)

# Fit and plot the double exponential decay curve
fit_and_plot_exponential_decay(axes[1], time_us, voltage)

plt.tight_layout()
plt.show()
