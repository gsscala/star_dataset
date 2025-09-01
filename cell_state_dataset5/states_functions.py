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
    return data.__dict__


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
    data["status"] = np.where((data["Line"] > 34), "Current Pulses", data["status"])

    return data


def plot_battery_data(data, start=None, end=None):
    """
    Plots current, voltage, and battery states in a single figure.

    Parameters:
        data (dict): cell data
        start (float, optional): Start time for the plot. Default=None.
        end (float, optional): End time for the plot. Default=None.
    """
    # Ensure the data is a DataFrame
    columns = ["Time", "Line", "I", "U", "status"]
    data = pd.DataFrame(data, columns=columns)

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
    # Ensure the data is a DataFrame
    columns = ["Time", "status"]
    data = pd.DataFrame(data, columns=columns)

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