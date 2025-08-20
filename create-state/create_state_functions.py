from scipy.io import loadmat
from os import listdir, makedirs
from json import dump
import matplotlib.pyplot as plt

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


def classify_current_state(data, eps=0.01):
    """
    Classify battery current state based on current (I) values.

    Parameters:
    data (dict): Dictionary containing battery data fields
    eps (float): Small tolerance to classify idle state

    Returns:
    list: List containing the current state for each timestep:
          1 = charging, 0 = idle, -1 = discharging
    """
    estado = []  # Store classified current states

    # Iterate through all current measurements
    for current in data["I"]:
        if -eps <= current <= eps:     # Idle state
            estado.append(0)
        elif current < -eps:           # Discharging
            estado.append(-1)
        else:                          # Charging
            estado.append(1)

    return estado


def process_all_cells(directory_name):
    """
    Process all MATLAB files in CU_Dynamic, classify states,
    and save the results as JSON files.

    Parameters:
    directory_name (str): Path to the CU_Dynamic folder
    """
    # Iterate through all check-up folders
    for checkup_name in sorted(listdir(directory_name)):
        if checkup_name == "failed and incomplete":  # Skip failed and incomplete folders
            continue

        # Iterate through all cells within the current check-up
        for cell in listdir(f"{directory_name}/{checkup_name}"):
            file_path = f"{directory_name}/{checkup_name}/{cell}"

            # Load MATLAB data for the current cell
            data = load_cell_data(file_path)

            # Classify battery current state for the entire timeline
            data["estado"] = classify_current_state(data)

            # Create output directory if it doesn't exist
            out_dir = f"Edited_dics/{checkup_name}"
            makedirs(out_dir, exist_ok=True)

            # Save classified states to a JSON file
            with open(f"{out_dir}/{cell[:-4]}.json", "w") as f:
                dump(data["estado"], f)


def plot_estado_over_time(data, estado):
    """
    Plot battery current state (estado) over time.

    Parameters:
    data (dict): Dictionary containing battery dataset fields
    estado (list): List containing the classified current state values

    Returns:
    None
    """
    plt.plot(data["Time"], estado, linewidth=1, color="blue")  # Plot estado vs time
    plt.xlabel("Time (s)")
    plt.ylabel("Estado")
    plt.grid(True)
    plt.show()