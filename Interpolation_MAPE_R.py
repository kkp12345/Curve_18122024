import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Data
battery_voltage = [4.50, 4.25, 4.00, 3.75, 3.50, 3.25, 3.00, 2.75, 2.50, 2.25, 2.00]
time = [44, 47, 52, 57, 64, 70, 81, 94, 110, 140, 200]

# Interpolation
interpolator = interp1d(time, battery_voltage, kind='cubic')  # Use cubic interpolation

# Generate interpolated points
interpolated_time = np.linspace(min(time), max(time), 500)
interpolated_voltage = interpolator(interpolated_time)

# Calculate MAPE
predicted_voltage = interpolator(time)
actual_voltage = battery_voltage
mape = np.mean(np.abs((np.array(actual_voltage) - predicted_voltage) / np.array(actual_voltage))) * 100

# Calculate R-squared
ss_total = np.sum((np.array(actual_voltage) - np.mean(actual_voltage))**2)
ss_residual = np.sum((np.array(actual_voltage) - predicted_voltage)**2)
r_squared = 1 - (ss_residual / ss_total)

# Print MAPE and R-squared
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R-squared: {r_squared:.4f}")

# Plot
plt.figure(figsize=(8, 6))
plt.plot(time, battery_voltage, 'o', label='Original Data', markersize=8)
plt.plot(interpolated_time, interpolated_voltage, '-', label='Cubic Interpolation')
plt.xlabel('Time (msec)', fontsize=12)
plt.ylabel('Battery Voltage (V)', fontsize=12)
plt.title('Battery Voltage vs. Time with Interpolation', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()
