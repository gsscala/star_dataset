import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.ticker as mticker
from os import listdir

def load_cell_data(file_path):
    """
    Load MATLAB data file (.mat) and return as a dictionary.

    Parameters:
    file_path (str): Path to the .mat file

    Returns:
    data.__dict__: Battery data as dictionary
    """
    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)["Dataset"]
    return data.__dict__  # Return dataset as dict

def plot_capacity_vs_time(time_data, capacity_data):
    """
    Plot capacity vs time for battery data.
    
    Parameters:
    time_data (array): Time values
    capacity_data (array): Capacity values
    """
    plt.plot(time_data, capacity_data)
    plt.ylabel("Capacity (Ah)")
    plt.xlabel("Time (t)")
    plt.title("Capacity vs Time")
    plt.tight_layout()
    plt.show()

def calculate_soh(capacities, nominal_capacity=2.5):
    """
    Calculate State of Health (SoH) from capacity values.
    
    Parameters:
    capacities (array): Capacity values
    nominal_capacity (float): Nominal capacity of the battery
    
    Returns:
    array: SoH values in percentage
    """
    return np.array(capacities) * 100 / nominal_capacity

def extract_capacity_data(directory_name, cell_id):
    """
    Extract capacity values from all check-ups for a specific cell.
    
    Parameters:
    directory_name (str): Name of the directory containing check-up data (CU_Dynamic in this case)
    cell_id (str): Internal ID of the battery cell (BW-VTC-xxx)
    
    Returns:
    tuple: (capacities, checkups) - arrays of capacity values and check-up numbers
    """
    capacities = [] # Store the capacity values
    checkups = [] # Store the check-up numbers
    
    # Iterate through all check-up folders
    for checkup_name in sorted(listdir(directory_name)): 
        if checkup_name == "failed and incomplete": # Skip failed and incomplete check-ups
            continue
            
        # Iterate through all cell files within the current check-up
        for cell in listdir(f"{directory_name}/{checkup_name}"):
            if cell.startswith(cell_id): # Process only files that match the target cell ID
                data = load_cell_data(f"{directory_name}/{checkup_name}/{cell}") # Load data

                # Calculate and store capacity of this check-up
                current_capacity = max(data["Ah"]) - min(data["Ah"]) 
                capacities.append(current_capacity) 

                # Extract and store check-up number
                checkups.append(int(checkup_name[2:5])) 
                
    return capacities, checkups

def plot_soh_vs_checkups(checkups, soh_values):
    """
    Plot SoH vs check-ups for battery cell.
    
    Parameters:
    checkups (array): Check-up numbers
    soh_values (array): SoH values in percentage
    """
    plt.title("SoH over time")
    plt.ylabel("SoH (%)")
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    plt.xlabel("Checkup")
    plt.xticks(checkups)
    plt.scatter(checkups, soh_values)
    plt.tight_layout()
    plt.show()