import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the double exponential model (approximating the given equation)
def double_exponential(t, a1, b1, a2, b2, C):
    return a1 * np.exp(b1 * t) + a2 * np.exp(b2 * t) + C

# Define the original generalized equation Vin = (0.5 * Vcc) / (1 - exp(-t_th / (RC)))
def generalized_model(t, Vcc, R, C, t_th):
    return (0.5 * Vcc) / (1 - np.exp(-t_th / (R * C)))

# Generate synthetic data for fitting (using the generalized equation as an example)
t_data = np.linspace(0, 10, 100)  # Time data
Vcc = 5.0  # Example supply voltage
R = 1.0    # Example resistance
C = 1.0    # Example capacitance
t_th = 2.0 # Example threshold time

# Generate the Vin data based on the generalized model
Vin_data = generalized_model(t_data, Vcc, R, C, t_th)

# Add some noise to simulate real data
noise = 0.1 * np.random.normal(size=t_data.shape)
Vin_data_noisy = Vin_data + noise

# Fit the double exponential model to the noisy data
popt, pcov = curve_fit(double_exponential, t_data, Vin_data_noisy, p0=[1, -1, 0.5, 0.1, 2])

# popt contains the best-fit parameters
a1_fit, b1_fit, a2_fit, b2_fit, C_fit = popt

# Print the fitted parameters
print(f"Fitted parameters:")
print(f"a1: {a1_fit:.3f}, b1: {b1_fit:.3f}, a2: {a2_fit:.3f}, b2: {b2_fit:.3f}, C: {C_fit:.3f}")

# Plot the data and the fitted curve
plt.plot(t_data, Vin_data_noisy, label="Noisy Data", linestyle='dashed', color='gray')
plt.plot(t_data, double_exponential(t_data, *popt), label="Fitted Double Exponential", color='red')
plt.legend()
plt.xlabel('Time (t)')
plt.ylabel('Vin (Voltage Input)')
plt.title('Double Exponential Fit to Generalized Model')
plt.show()
