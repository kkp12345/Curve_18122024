import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Global variables to store file path and data
file_path = None
data = None
filtered_data = None

# Function to load data file
def load_file():
    global file_path, data
    file_path = filedialog.askopenfilename()
    if file_path:
        data = pd.read_csv(file_path)
        plot_data()
        root.title(f"Data Plot - {file_path}")

# Function to save the filtered data
def save_filtered_data():
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if save_path and filtered_data is not None:
        filtered_data.to_csv(save_path, index=False)
        print(f"Filtered data saved to {save_path}")

# Function to apply median filtering to the LM35 data
def filter_data():
    global filtered_data
    if data is None:
        print("No data loaded.")
        return
    
    lm35_values = data['LM35'].values.copy()
    freq_values = data['Frequency'].values.copy()
    lm35_values2 = data['LM35'].values.copy()
    freq_values2 = data['Frequency'].values.copy()
    
    # Apply median filter with a window of 5 points
    for i in range(2, len(lm35_values) - 2):
    # for i in range(1, 5):
        windowL = lm35_values[i-2:i+3]
        a=lm35_values[i]
        b=np.median(windowL)
        if a > b:
            print(a,b)
            lm35_values2[i] = (lm35_values[i-1]+lm35_values[i+1])/2

        windowF = freq_values[i-1:i+2]
        c=freq_values[i]
        d=np.median(windowF)
        if c > d:
            print(c,d)
            freq_values2[i] = (freq_values[i-1]+freq_values[i+1])/2
    
    # Create a new DataFrame for filtered data
    try:
        filtered_data = data.copy()
        filtered_data['LM35'] = lm35_values2
        filtered_data['Frequency'] = freq_values2
        print("Filtered Data DataFrame:", filtered_data.head())
    except Exception as e:
        print("Error:", e)


    filtered_dataL = data.copy()
    filtered_dataF = data.copy()
    filtered_dataL['LM35'] = lm35_values2
    filtered_dataF['Frequency'] = freq_values2
    
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if save_path and filtered_data is not None:
        filtered_data.to_csv(save_path, index=False)
        print(f"Filtered data saved to {save_path}")



    # Plot the filtered data
    plt.plot(filtered_dataL['LM35'], filtered_dataF['Frequency'], 'o-', label='Filtered Data', color='red')
    plt.title(f"Original vs Filtered Data - {file_path}")
    plt.xlabel('LM35 Temperature')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Function to plot the original data
def plot_data():
    if data is None:
        print("No data loaded.")
        return

    plt.plot(data['LM35'], data['Frequency'], 'o-', label='Original Data', color='blue')
    plt.title(f"Original Data - {file_path}")
    plt.xlabel('LM35 Temperature')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# Function to save the plot as a JPEG image
def save_plot():
    save_path = filedialog.asksaveasfilename(defaultextension=".jpeg", filetypes=[("JPEG files", "*.jpeg")])
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved as JPEG at {save_path}")

# Create the main window
root = tk.Tk()
root.title("Data Plotter")

# Add buttons for loading and saving files, and for filtering
load_button = tk.Button(root, text="Load File", command=load_file)
load_button.pack()

filter_button = tk.Button(root, text="Filter", command=filter_data)
filter_button.pack()

save_button = tk.Button(root, text="Save Plot", command=save_plot)
save_button.pack()

save_filtered_button = tk.Button(root, text="Save Filtered Data", command=save_filtered_data)
save_filtered_button.pack()

# Run the application
root.mainloop()
