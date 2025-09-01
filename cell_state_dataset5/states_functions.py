import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pandas.api.types import CategoricalDtype

def load_cell_data(file_path):
    """
    Load the battery dataset from a .mat file.

    Parameters:
        file_path (str): Path to the .mat file.

    Returns:
        dict: Dictionary containing the dataset fields.
    """
    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)["Dataset"]
    data = data.__dict__
    columns = ["Time", "Line", "I", "U", "status", "T1"]
    data = pd.DataFrame(data, columns=columns)

    return data


def process_states(path):
    """
    Process the battery dataset and assign operational states based on the 'Line' column.

    Parameters:
        path (str): Path to the .mat dataset file.

    Returns:
        dict: Dataset including the computed 'status' column.
    """
    data = load_cell_data(path)

    # Define operational states based on 'Line' value ranges
    data["status"] = np.where(data["Line"] <= 13, "Rest", "")
    data["status"] = np.where((data["Line"] > 13) & (data["Line"] < 15), "CC Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 14) & (data["Line"] < 16), "Rest", data["status"])
    data["status"] = np.where((data["Line"] > 15) & (data["Line"] < 17), "CC Charge", data["status"])
    data["status"] = np.where((data["Line"] > 16) & (data["Line"] < 19), "CV Charge", data["status"])
    data["status"] = np.where((data["Line"] > 19) & (data["Line"] < 21), "CC Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 20) & (data["Line"] < 22), "CV Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 21) & (data["Line"] < 24), "CC Charge", data["status"])
    data["status"] = np.where((data["Line"] > 23) & (data["Line"] < 26), "CV Charge", data["status"])
    data["status"] = np.where((data["Line"] > 26) & (data["Line"] < 28), "CC Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 27) & (data["Line"] < 33), "CV Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 32) & (data["Line"] < 35), "CC Charge", data["status"])
    data["status"] = np.where((data["Line"] > 34) & (data["Line"] < 36), "CV Charge", data["status"])
    data["status"] = np.where((data["Line"] > 35) & (data["Line"] < 37), "Rest", data["status"])
    data["status"] = np.where((data["Line"] > 37) & (data["Line"] < 42), "Current Pulses", data["status"])
    data["status"] = np.where((data["Line"] > 41) & (data["Line"] < 43), "CC Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 42) & (data["Line"] < 44), "Rest", data["status"])
    data["status"] = np.where((data["Line"] > 43) & (data["Line"] < 49), "Current Pulses", data["status"])
    data["status"] = np.where((data["Line"] > 48) & (data["Line"] < 50), "CC Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 49) & (data["Line"] < 51), "Rest", data["status"])
    data["status"] = np.where((data["Line"] > 50) & (data["Line"] < 56), "Current Pulses", data["status"])
    data["status"] = np.where((data["Line"] > 55) & (data["Line"] < 57), "CC Charge", data["status"])
    data["status"] = np.where((data["Line"] > 56), "Rest", data["status"])
    return data

def analyze_metrics_by_status_period(data):
    """
    Analyze and compute metrics for each continuous period of a specific battery status.
    Parameters:
        data (dict): cell data 
    Returns:
        pd.DataFrame: DataFrame containing metrics for each status period.
    """

    # Identify changes in status to segment the data
    data["status_change"] = (data["status"] != data["status"].shift()).cumsum()

    # Group by status change segments
    grouped = data.groupby("status_change")

    results = []
    for _, g in grouped:

        subset = g.iloc[1:-1]  # Exclude first and last rows of each group

        status = subset["status"].iloc[0]
        start_time = subset["Time"].iloc[0]
        end_time = subset["Time"].iloc[-1]
        duration = end_time - start_time

        avg_current = subset["I"].mean()
        median_current = subset["I"].median()
        std_current = subset["I"].std()

        avg_tension = subset["U"].mean()
        median_tension = subset["U"].median()
        std_tension = subset["U"].std()

        avg_temperature = subset["T1"].mean()
        median_temperature = subset["T1"].median()
        std_temperature = subset["T1"].std()

        results.append({
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "avg_current": avg_current,
            #"median_current": median_current,
            "std_current": std_current,
            "avg_tension": avg_tension,
            #"median_tension": median_tension,
            "std_tension": std_tension,
            "avg_temperature": avg_temperature,
            #"median_temperature": median_temperature,
            "std_temperature": std_temperature
        })

    return pd.DataFrame(results)

def plot_battery_data(data, start=None, end=None):
    """
    Plots current, voltage, and battery states in a single figure.

    Parameters:
        data (dict): cell data
        start (float, optional): Start time for the plot. Default=None.
        end (float, optional): End time for the plot. Default=None.
    """

    # Filter data based on start and end time if provided
    if start is not None:
        data = data[data["Time"] >= start]
    if end is not None:
        data = data[data["Time"] <= end]

    # Fixed status order (user cannot change)
    fixed_order = ["CV Discharge", "CC Discharge", "Rest", "CV Charge", "CC Charge", "Current Pulses"]
    unique_statuses = [s for s in fixed_order if s in data["status"].unique()]

    # Convert status column to categorical type with fixed order
    cat_type = CategoricalDtype(categories=unique_statuses, ordered=True)
    data["status"] = data["status"].astype(cat_type)
    data["status_num"] = data["status"].cat.codes

    # Detect status transition points for visualization
    data["Change"] = data["Line"].diff().fillna(0)
    transitions = data[data["Change"] != 0]

    # Create figure and main axis (left)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left y-AXIS
    ax1.plot(data["Time"], data["status_num"], color="purple", linestyle="--", label="Status")
    ax1.set_yticks(range(len(unique_statuses)))
    ax1.set_yticklabels(unique_statuses)
    ax1.set_ylabel("Battery Status")
    ax1.grid()

    # Right y-axis
    ax2 = ax1.twinx()
    ax2.plot(data["Time"], data["I"], label="Current [A]", color="blue")
    ax2.plot(data["Time"], data["U"], label="Voltage [V]", color="green")
    ax2.scatter(transitions["Time"], transitions["I"], color="red", marker="o", s=60, label="State Changes")
    ax2.set_ylabel("Current [A] / Voltage [V]")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    # Final settings
    plt.title("Battery Operating States")
    plt.xlabel("Time [s]")
    plt.show()

def plot_battery_status(data, start=None, end=None):
    """
    Plots only the battery operating status over time.

    Parameters:
        data (dict): cell data
        start (float, optional): Start time for the plot. Default=None.
        end (float, optional): End time for the plot. Default=None.
    """

    # Filter by start and end time if provided
    if start is not None:
        data = data[data["Time"] >= start]
    if end is not None:
        data = data[data["Time"] <= end]

    # Fixed status order
    fixed_order = ["CV Discharge", "CC Discharge", "Rest", "CV Charge", "CC Charge", "Current Pulses"]
    unique_statuses = [s for s in fixed_order if s in data["status"].unique()]

    # Convert status to categorical type with fixed order
    cat_type = CategoricalDtype(categories=unique_statuses, ordered=True)
    data["status"] = data["status"].astype(cat_type)
    data["status_num"] = data["status"].cat.codes

    # Create figure
    plt.figure(figsize=(12, 5))

    # Plot status curve
    plt.plot(data["Time"], data["status_num"], label="Battery Status")

    # Set custom Y-axis labels
    plt.yticks(range(len(unique_statuses)), unique_statuses)

    # Labels and title
    plt.title("Battery Operating Status Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Battery Status")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()