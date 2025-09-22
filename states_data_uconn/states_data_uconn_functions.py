import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import h5py

def load_data(src):
    with h5py.File(src, "r") as f:
        keys = list(f.keys())
        raw = f[keys[0]][:]  

    # Convert to text
    text = raw.tobytes().decode("utf-8")

    # Load as json
    data_json = json.loads(text)

    df = pd.DataFrame(data_json)

    return df

def plot_segment(n_period, df, results):
    """
    Plot raw Current, Voltage, and State for a specific segment.
    
    Parameters:
    n_period : int
        Segment index in 'results' to plot.
    df : DataFrame
        Raw measurement data with columns "Time (s)", "Voltage (V)", "Current (A)", "State".
    results : DataFrame
        Summary DataFrame with 'start_time', 'end_time', and 'status'.
    """
    
    # Select the segment from 'results'
    segment = results.iloc[n_period]
    
    # Segment time period
    start_period = segment["start_time"]
    end_period = segment["end_time"]
    
    # Filter raw data for the segment
    period_df = df[(df["Time (s)"] >= start_period) & (df["Time (s)"] <= end_period)]
    
    # Prepare current and voltage values
    current = period_df["Current (A)"]
    voltage = period_df["Voltage (V)"]
    
    # Plot current and voltage on the same scale
    plt.figure(figsize=(12, 6))
    plt.plot(period_df["Time (s)"], current, label="Current (A)", color="blue")
    plt.plot(period_df["Time (s)"], voltage, label="Voltage (V)", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title(f"Current and Voltage for segment {segment['status']}")
    plt.legend()
    plt.show()
    
    # Plot state as step plot
    plt.figure(figsize=(12, 2))
    plt.step(period_df["Time (s)"], period_df["State"], where='post', color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("State")
    plt.title(f"State for segment {segment['status']}")
    plt.show()
