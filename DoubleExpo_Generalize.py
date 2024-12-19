import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---- PART 1: Voltage from Formula ---- #
def calculate_voltage(t_th, R, C, Vcc):
    """
    Calculate Vin using the given formula.
    :param t_th: Time in seconds (array or scalar)
    :param R: Resistance in ohms
    :param C: Capacitance in farads
    :param Vcc: Supply voltage
    :return: Calculated Vin
    """
    return (0.5 * Vcc) / (1 - np.exp(-t_th / (R * C)))

def plot_voltage_from_formula(ax, R, C, Vcc, t_th_range):
    """
    Plot Vin as a function of time t_th for given R, C, and Vcc on the provided axis.
    """
    t_th = np.linspace(t_th_range[0], t_th_range[1], 500)  # Generate time values
    V_in = calculate_voltage(t_th, R, C, Vcc)  # Calculate voltage
    
    ax.plot(t_th, V_in, label=f"Formula: R={R}Ω, C={C}F, Vcc={Vcc}V", color="blue")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage vs. Time (Formula-based)")
    ax.legend()
    ax.grid(True)

# ---- PART 2: Double Exponential Fit ---- #
# Provided data
voltage = np.array([4.48, 4.27, 4.00, 3.75, 3.48, 3.23, 3.00, 2.75, 2.50, 2.25, 2.00])
time_us = np.array([44000, 47000, 52000, 57000, 64000, 70000, 81000, 94000, 110000, 140000, 200000])
time_us_scaled = time_us / 1e6  # Convert time to seconds

# Define the double exponential decay model
def double_exponential_decay(x, a1, b1, a2, b2, c):
    return a1 * np.exp(b1 * x) + a2 * np.exp(b2 * x) + c

# Fit the model
lower_bounds = [0, -50, 0, -50, 0]
upper_bounds = [10, 0, 10, 0, 5]
initial_guesses = [5, -1, 1, -0.1, 2]

params, _ = curve_fit(double_exponential_decay, time_us_scaled, voltage, p0=initial_guesses, bounds=(lower_bounds, upper_bounds))
a1, b1, a2, b2, c = params

def plot_voltage_from_fit(ax, time_us_scaled, voltage, params):
    """
    Plot observed data and fitted curve on the provided axis.
    """
    time_fit_scaled = np.linspace(min(time_us_scaled), max(time_us_scaled), 1000)
    voltage_fit = double_exponential_decay(time_fit_scaled, *params)

    ax.scatter(time_us_scaled, voltage, color='blue', label="Observed Data")
    ax.plot(time_fit_scaled, voltage_fit, color='red', label="Fitted Double Exponential Model")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Battery Voltage vs. Time (Exponential Fit)")
    ax.legend()
    ax.grid(True)

# ---- Combined Plot ---- #
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot for Formula-based voltage
R = 1000  # Resistance in ohms
C = 0.001  # Capacitance in farads
Vcc = 5  # Supply voltage in volts
t_th_range = (0.001, 0.01)  # Time range in seconds
plot_voltage_from_formula(axes[0], R, C, Vcc, t_th_range)

# Plot for Exponential Decay Fit
plot_voltage_from_fit(axes[1], time_us_scaled, voltage, params)

# Show both plots
plt.tight_layout()
plt.show()
