import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.ticker as mticker
import os
import re

def select_cells_ids():
    """
    Ask the user to enter battery cell IDs for analysis.

    Returns:
    list: List of selected cell IDs (strings)
    """

    cell_ids = []

    while True:
        cell_id = input("Cell ID: ").strip()
        if cell_id == "0":
            break
        elif not cell_id.isdigit():
            print("Please enter only numeric values!")
        else:
            cell_ids.append(cell_id)

    return cell_ids

def find_cell_check_ups_files(cell_id):
    """
    Find all check-up files for a specific battery cell.

    Parameters:
    cell_id (str): Internal ID of the battery cell (xxx from 'BW-VTC-xxx').

    Returns:
    list: Sorted list of paths to .mat files (check-up files).
    """
    paths = []

    # Walk through all folders inside "Data_publication"
    for root, dirs, files in os.walk("Data_publication"):
        # Filter out unwanted folders
        dirs[:] = [
            d for d in dirs 
            if 'failed and incomplete' not in d.lower()
            and d not in ['CYC_Dynamic', 'CYC_Cyclic']
        ]

        # Check if file matches the requested battery ID
        paths.extend(
            os.path.join(root, f) 
            for f in files 
            if f.startswith(f'BW-VTC-{cell_id}')
        )

    return sorted(paths)

def get_valid_cells_and_checkups():
    """
    Select cells, find their check-up files, and filter out invalid IDs.

    Returns:
    tuple:
        valid_cell_ids (list): List of valid battery IDs.
        checkups (list of list): For each valid cell, a list of check-up file paths.
    """
    
    cell_ids = select_cells_ids() # Ask user to input cell IDs
    valid_cell_ids = []
    checkups = []

    # Iterate through all selected cell IDs
    for cell_id in cell_ids:
        paths = find_cell_check_ups_files(cell_id) # Find all check-up files for the current cell

        # If files exist, store the cell ID and its check-ups
        if paths: 
            valid_cell_ids.append(cell_id)
            checkups.append(paths)
        else:
            print(f"No check-up files found for cell {cell_id}.")

    return valid_cell_ids, checkups # Return only valid cell IDs and their check-up paths

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

def plot_initial_and_final_capacity(checkups, cell_ids):
    """
    Plot capacity vs time for the initial and final check-ups of multiple battery cells.
    Each cell gets an individual plot showing both curves.

    Parameters:
    checkups (list of list): For each cell, a list of paths to .mat files containing check-up data.
    cell_ids (list of str): List of internal battery IDs (xxx from 'BW-VTC-xxx').

    Returns:
    None
    """

    # Iterate through all selected cells
    for i, paths in enumerate(checkups):
        # Load first and last check-up data for the current cell
        initial_data = load_cell_data(paths[0])
        final_data = load_cell_data(paths[-1])

        # Extract cell ID if not provided explicitly (from filename)
        file_name = os.path.basename(paths[0])
        match = re.search(r'BW-VTC-(\d+)', file_name)
        cell_id = match.group(1) if match else cell_ids[i]

        # Plot both curves on the same chart
        plt.figure(figsize=(8, 5))
        plt.plot(initial_data["Time"], initial_data["Ah"], label="Initial check-up", color="blue")
        plt.plot(final_data["Time"], final_data["Ah"], label="Final check-up", color="red")

        # Configure chart appearance
        plt.xlabel("Time (t)")
        plt.ylabel("Capacity (Ah)")
        plt.title(f"Initial vs Final Capacity (cell {cell_id})")
        plt.legend()
        plt.tight_layout()
        plt.show()

def calculate_soh(capacity_values, cell_ids, nominal_capacity=2.5):
    """
    Calculate State of Health (SoH) from capacity values for multiple cells.

    Parameters:
    capacity_values (list of arrays): List of capacity arrays for each cell
    cell_ids (list): List of cell identifiers
    nominal_capacity (float): Nominal capacity of the battery

    Returns:
    list: SoH values in percentage for each cell
    """

    soh_percentages = []
    
    # Iterate through each cell
    for i in range(len(cell_ids)):
        cell_capacities = capacity_values[i] # Get capacity values for current cell
        soh_array = np.array(cell_capacities) * 100 / nominal_capacity # Calculate SoH as percentage
        soh_percentages.append(soh_array) # Append the SoH array of the current cell to the main list
    
    return soh_percentages # Return the list of SoH arrays for all cells

def extract_all_capacity_data(checkups, cell_ids):
    """
    Extract capacity values and check-up numbers for multiple battery cells.

    Parameters:
    checkups (list of list): For each cell, a list of paths to .mat files containing check-up data.
    cell_ids (list of str): List of internal battery IDs (xxx from 'BW-VTC-xxx').

    Returns:
    tuple: (capacity_values, checkup_cycles)
        capacity_values (list of list): For each cell, capacity values from all check-ups.
        checkup_cycles (list of list): For each cell, corresponding check-up numbers.
    """
    capacity_values = []   # Store capacity arrays for each cell
    checkup_cycles = []    # Store check-up numbers for each cell

    # Iterate through all selected cells
    for i, paths in enumerate(checkups):
        capacities = []        
        checkups_numbers = [] 

        # Iterate through all check-up files for the current cell
        for path in paths:
            data = load_cell_data(path)  # Load data

            current_capacity = np.ptp(data["Ah"]) # ptp = max(Ah) - min(Ah)
            capacities.append(current_capacity)  # Save capacity value

            # Extract check-up number using regex: finds 'CU###' in the file path
            match = re.search(r'/CU(\d+)(?=[_/])', path)
            if match:
                checkups_numbers.append(int(match.group(1)))  # Save check-up number

        # Store processed data for this cell
        capacity_values.append(capacities)
        checkup_cycles.append(checkups_numbers)

        # Log progress for the current cell
        print(f"Cell {cell_ids[i]}: processed {len(checkups_numbers)} check-up cycles")

    return capacity_values, checkup_cycles


def plot_multiple_soh(cell_ids, checkup_cycles, soh_values):
    """
    Plot SoH (%) across check-ups for multiple battery cells on a single chart.

    Each cell is represented by a line showing its SoH trend across check-ups.

    Parameters:
    cell_ids (list of str): List of internal battery IDs (xxx from 'BW-VTC-xxx').
    checkup_cycles (list of array-like): For each cell, the sequence of check-up numbers.
    soh_values (list of array-like): For each cell, the corresponding SoH (%) values.

    Returns:
    None
    """

    # Create a new figure for the combined SoH plot
    plt.figure(figsize=(8, 5))

    # Iterate through all selected cells
    for i, cid in enumerate(cell_ids):
        # Plot SoH values versus check-up numbers for the current cell
        plt.plot(
            checkup_cycles[i],   # x-axis: check-up numbers
            soh_values[i],       # y-axis: SoH percentages
            marker="o",          # mark each data point
            label=f"Cell {cid}"  # label for legend
        )

    # Configure chart appearance
    plt.xlabel("Check-up") 
    plt.ylabel("SoH (%)")   
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    plt.title("Battery SoH Degradation")
    plt.locator_params(axis="x", nbins=10)
    plt.legend()
    plt.tight_layout()
    plt.show()