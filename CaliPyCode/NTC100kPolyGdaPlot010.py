import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Initialize global variables for the data
data = None
filtered_data = None
filename = ""

def load_file():
    global data, filename
    filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filename:
        data = pd.read_csv(filename)
        plot_data()

def save_plot():
    if data is not None:
        save_filename = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
        if save_filename:
            plt.savefig(save_filename)
            print(f"Plot saved as {save_filename}")

def save_filtered_data():
    global filtered_data
    if filtered_data is not None:
        save_filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_filename:
            filtered_data.to_csv(save_filename, index=False)
            print(f"Filtered data saved as {save_filename}")

def plot_data(filtered=False):
    plt.figure(figsize=(10, 6))
    
    # Plot original data
    plt.plot(data['LM35'], data['Frequency'], 'o-', label='Original Data', color='blue')

    if filtered:
        # Plot filtered data
        plt.plot(filtered_data['LM35'], filtered_data['Frequency'], 'o-', label='Filtered Data', color='red')

    plt.xlabel('LM35 (Temperature)')
    plt.ylabel('Frequency')
    plt.title(f"LM35 vs Frequency\n{filename.split('/')[-1]}")  # Add file name to title
    plt.legend()
    plt.show()

def filter_data():
    global filtered_data
    if data is not None:
        threshold_min = float(threshold_min_entry.get())
        threshold_max = float(threshold_max_entry.get())

        # Apply threshold-based filtering
        filtered_data = data[(data['LM35'] >= threshold_min) & (data['LM35'] <= threshold_max)]

        # Plot the filtered data
        plot_data(filtered=True)

# Create the main window
root = tk.Tk()
root.title("LM35 vs Frequency Plotter")

# Load file button
load_button = tk.Button(root, text="Load CSV", command=load_file)
load_button.pack()

# Save plot button
save_plot_button = tk.Button(root, text="Save Plot", command=save_plot)
save_plot_button.pack()

# Threshold entries and labels
threshold_min_label = tk.Label(root, text="LM35 Min Threshold")
threshold_min_label.pack()
threshold_min_entry = tk.Entry(root)
threshold_min_entry.pack()

threshold_max_label = tk.Label(root, text="LM35 Max Threshold")
threshold_max_label.pack()
threshold_max_entry = tk.Entry(root)
threshold_max_entry.pack()

# Filter button
filter_button = tk.Button(root, text="Filter", command=filter_data)
filter_button.pack()

# Save filtered data button
save_filtered_button = tk.Button(root, text="Save Filtered CSV", command=save_filtered_data)
save_filtered_button.pack()

# Run the application
root.mainloop()
