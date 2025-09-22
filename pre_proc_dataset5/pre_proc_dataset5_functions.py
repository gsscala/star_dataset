import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import json
import h5py
import os

def load_h5(path):
    """
    Load h5 file and convert it to a pandas DataFrame.

    Parameters:
        path (str): Path to the .mat dataset file.
    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """

    with h5py.File(path, "r") as f:
        key = list(f.keys())[0]
        raw = f[key][:]  # uint8 array

    # Convert to text
    text = raw.tobytes().decode("utf-8")

    # Load as json
    data_json = json.loads(text)

    # Extract relevant fields
    data = {
        "Time": data_json["Time"],
        "I": data_json["I"],
        "U": data_json["U"],
        "Line": data_json["Line"]
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)   

    return df 

def plot_battery_data(data, n_period=None, metrics=None):
    """
    Plots status (top), tension/voltage (middle), and current + state changes (bottom).

    Parameters:
        data (pd.DataFrame): cell data
        n_period (int, optional): Specific period number to focus on. Default=None.
        metrics (pd.DataFrame, optional): DataFrame containing metrics for each status period. Default=None.
    """
    data = data.copy()

    # --- determine start/end from metrics if provided (robust: try loc then iloc) ---
    start, end = None, None
    if n_period is not None and metrics is not None:
        row = None
        try:
            row = metrics.loc[n_period]
        except Exception:
            try:
                row = metrics.iloc[int(n_period)]
            except Exception:
                row = None
        if row is not None:
            try:
                start = float(row["start_time"]) - 0.25
                end = float(row["end_time"]) + 0.25
            except Exception:
                start, end = None, None

    # Filter data based on start and end time if provided
    if start is not None:
        data = data[data["Time"] >= start]
    if end is not None:
        data = data[data["Time"] <= end]

    # fixed order + fallback
    fixed_order = ["CV Discharge", "CC Discharge", "Rest", "CV Charge", "CC Charge", "Current Pulses"]
    present = list(data["status"].dropna().unique())
    unique_statuses = [s for s in fixed_order if s in present]
    if not unique_statuses:
        unique_statuses = [s for s in present if pd.notna(s)]

    if unique_statuses:
        cat_type = CategoricalDtype(categories=unique_statuses, ordered=True)
        data["status"] = data["status"].astype(cat_type)
        data["status_num"] = data["status"].cat.codes
    else:
        data["status_num"] = 0

    # transitions by Line changes
    data["Change"] = data["Line"].diff().fillna(0)
    transitions = data[data["Change"] != 0]

    # three stacked subplots: status, tension, current
    fig, (ax_status, ax_tension, ax_current) = plt.subplots(
        3, 1, sharex=True, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 1, 2]}
    )

    # top: status
    if unique_statuses:
        ax_status.plot(data["Time"], data["status_num"], color="purple", linestyle="--")
        ax_status.set_yticks(range(len(unique_statuses)))
        ax_status.set_yticklabels(unique_statuses)
    else:
        ax_status.text(0.5, 0.5, "No status data", ha="center", va="center")
    ax_status.set_ylabel("Status")
    ax_status.grid(axis="y", linestyle="--", alpha=0.5)

    # middle: tension / voltage
    ax_tension.plot(data["Time"], data["U"], color="green", label="Voltage [V]")
    ax_tension.set_ylabel("Voltage [V]")
    ax_tension.grid(True, linestyle="--", alpha=0.4)
    ax_tension.legend(loc="best")

    # bottom: current and state-changes
    ax_current.plot(data["Time"], data["I"], color="blue", label="Current [A]")
    if not transitions.empty:
        ax_current.scatter(transitions["Time"], transitions["I"], color="red", s=40, marker="o", label="State Changes")
    ax_current.set_ylabel("Current [A]")
    ax_current.set_xlabel("Time")
    ax_current.grid(True, linestyle="--", alpha=0.4)
    ax_current.legend(loc="best")

    ax_status.set_title("Battery Operating States / Signals")
    plt.tight_layout()
    plt.show()

def process_states(data):
    """
    Process the battery dataset and assign operational states based on the 'Line' column.

    Parameters:
        path (str): Path to the .mat dataset file.

    Returns:
        dict: Dataset including the computed 'status' column.
    """

    # Define operational states based on 'Line' value ranges
    data["status"] = np.where(data["Line"] <= 13, "Rest", "")
    data["status"] = np.where((data["Line"] > 13) & (data["Line"] < 15), "CC Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 14) & (data["Line"] < 16), "Rest", data["status"])
    data["status"] = np.where((data["Line"] > 15) & (data["Line"] < 17), "CC Charge", data["status"])
    data["status"] = np.where((data["Line"] > 16) & (data["Line"] < 19), "CV Charge", data["status"])
    data["status"] = np.where((data["Line"] > 19) & (data["Line"] < 21), "CC Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 20) & (data["Line"] < 22), "CV Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 21) & (data["Line"] < 23), "Rest", data["status"])
    data["status"] = np.where((data["Line"] > 22) & (data["Line"] < 24), "CC Charge", data["status"])
    data["status"] = np.where((data["Line"] > 23) & (data["Line"] < 26), "CV Charge", data["status"])
    data["status"] = np.where((data["Line"] > 26) & (data["Line"] < 28), "CC Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 27) & (data["Line"] < 33), "CV Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 32) & (data["Line"] < 34), "Rest", data["status"])
    data["status"] = np.where((data["Line"] > 33) & (data["Line"] < 35), "CC Charge", data["status"])
    data["status"] = np.where((data["Line"] > 34) & (data["Line"] < 36), "CV Charge", data["status"])
    data["status"] = np.where((data["Line"] > 35) & (data["Line"] < 37), "Rest", data["status"])
    data["status"] = np.where((data["Line"] > 37) & (data["Line"] < 42), "Current Pulses", data["status"])
    data["status"] = np.where((data["Line"] > 41) & (data["Line"] < 43), "CC Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 42) & (data["Line"] < 44), "Rest", data["status"])
    data["status"] = np.where((data["Line"] > 43) & (data["Line"] < 49), "Current Pulses", data["status"])
    data["status"] = np.where((data["Line"] > 48) & (data["Line"] < 50), "CC Discharge", data["status"])
    data["status"] = np.where((data["Line"] > 49) & (data["Line"] < 51), "Rest", data["status"])
    data["status"] = np.where((data["Line"] > 50) & (data["Line"] < 56), "Current Pulses", data["status"])
    data["status"] = np.where((data["Line"] > 55), "Rest", data["status"])
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

        subset = g.iloc[3:-3]

        if subset.empty or subset["status"].isna().all():
            continue

        status = subset["status"].iloc[0]
        start_time = subset["Time"].iloc[0]
        end_time = subset["Time"].iloc[-1]
        duration = end_time - start_time

        avg_current = subset["I"].mean()
        std_current = subset["I"].std()

        avg_tension = subset["U"].mean()
        std_tension = subset["U"].std()

        results.append({
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "avg_current": avg_current,
            "std_current": std_current,
            "avg_tension": avg_tension,
            "std_tension": std_tension
        })

    return pd.DataFrame(results)