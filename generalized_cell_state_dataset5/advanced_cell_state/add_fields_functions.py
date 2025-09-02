"""
Advanced Cell State Analysis - Utility Functions

This module provides utility functions for processing battery test data and creating
advanced state classifications. It includes functions for:

1. Derivative calculation with smoothing for time series data
2. Noise-tolerant sign classification for derivative analysis  
3. JSON serialization of numpy data structures

These functions are used to analyze current and voltage derivatives to classify
battery states (constant current, constant voltage, dynamic, rest) with higher
precision than basic command-based classification.

Functions:
    - derivative(): Calculate smoothed derivatives of time series data
    - relaxed_sign(): Classify values as positive/negative/zero with threshold
    - make_json_serializable(): Convert numpy objects to JSON-compatible types

Author: Battery Analysis Team
Purpose: Advanced battery state classification for degradation studies
"""

# Import required libraries
import numpy as np

def derivative(library: list, pool: int) -> list:
    """
    Calculate the derivative of a time series using a sliding window approach.
    
    This function computes the derivative at each point by calculating the slope
    between two points symmetrically positioned around the current point within
    a specified window (pool). This provides smoothing and reduces noise.
    
    Args:
        library (list): Input time series data (e.g., current or voltage measurements)
        pool (int): Half-width of the sliding window for derivative calculation
                   (total window size = 2*pool + 1)
    
    Returns:
        list: Derivative values for each point in the input series
    
    Example:
        If pool=2, for point at index i, the derivative is calculated using
        points from max(i-2, 0) to min(i+2, len-1)
    """
    n = len(library)
    derivatives = []
    
    # Calculate derivative for each point in the series
    for x in range(n):
        # Define the window boundaries around current point
        x1 = max(x - pool, 0)        # Left boundary (don't go below 0)
        x2 = min(x + pool, n - 1)    # Right boundary (don't exceed array length)
        
        # Calculate slope (derivative) between the boundary points
        # This gives us the average rate of change in the local window
        variation = (library[x2] - library[x1]) / (x2 - x1)
        derivatives.append(variation)
        
    return derivatives

def relaxed_sign(num: float, eps: float) -> int:
    """
    Determine the sign of a number with a tolerance threshold (dead zone).
    
    This function classifies a number as positive, negative, or approximately zero
    based on a threshold value. This is useful for noisy data where small values
    near zero should be treated as zero rather than having a sign.
    
    Args:
        num (float): Input number to classify
        eps (float): Threshold value - numbers with absolute value <= eps are considered zero
    
    Returns:
        int: 
            1 if num > eps (significantly positive)
            -1 if num < -eps (significantly negative)  
            0 if -eps <= num <= eps (approximately zero)
    
    Example:
        relaxed_sign(0.1, 0.05) returns 1 (positive)
        relaxed_sign(-0.1, 0.05) returns -1 (negative)
        relaxed_sign(0.02, 0.05) returns 0 (approximately zero)
    """
    if num > eps:
        return 1        # Significantly positive
    if (num < -eps):
        return -1       # Significantly negative
    return 0            # Approximately zero (within threshold)

def make_json_serializable(obj):
    """
    Recursively convert numpy objects and arrays to JSON-serializable Python types.
    
    This function handles the conversion of numpy data types that cannot be directly
    serialized to JSON format. It recursively processes nested structures like
    dictionaries, lists, and arrays to ensure all elements are JSON-compatible.
    
    Args:
        obj: Any object that may contain numpy types (dict, list, array, scalar, etc.)
    
    Returns:
        JSON-serializable version of the input object with all numpy types converted
        to standard Python types (int, float, bool, list, dict)
    
    Supported conversions:
        - numpy arrays -> Python lists (recursive)
        - numpy integers -> Python int
        - numpy floats -> Python float  
        - numpy booleans -> Python bool
        - numpy generic types -> Python equivalents via .item()
        - Dictionaries and lists -> recursively processed
    """
    # Handle dictionaries by recursively processing all key-value pairs
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    # Convert numpy arrays to lists and process contents recursively
    if isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    
    # Handle lists and tuples by recursively processing all elements
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    
    # Convert numpy integer types to standard Python int
    if isinstance(obj, (np.integer,)):
        return int(obj)
    
    # Convert numpy floating point types to standard Python float
    if isinstance(obj, (np.floating,)):
        return float(obj)
    
    # Convert numpy boolean types to standard Python bool
    if isinstance(obj, np.bool_):
        return bool(obj)
    
    # Handle other numpy scalar types by extracting the Python equivalent
    if isinstance(obj, np.generic):
        return obj.item()
    
    # Return object unchanged if it's already JSON-serializable
    return obj

def find_state(dataset:dict, eps:float):
    """
    Classify battery states for each checkup using derivative analysis and segment smoothing.
    
    This function performs advanced state classification by:
    1. Calculating current and voltage derivatives
    2. Classifying states based on derivative signs and commands
    3. Smoothing classifications within segments (constant Line values)
    
    Args:
        dataset (dict): Battery data organized by checkup names
        eps (float): Threshold for relaxed sign classification of derivatives
    
    Returns:
        dict: Advanced state classifications for each checkup
    """
    advanced_state = {}
    
    # Process each checkup (CU000, CU001, etc.) independently
    for checkup_name in dataset.keys():
        advanced_state[checkup_name] = []
        
        # Calculate derivatives of current and voltage with smoothing (window size = 5)
        # Apply relaxed_sign to classify derivatives as positive(1), negative(-1), or zero(0)
        currents = np.vectorize(lambda x: relaxed_sign(x, eps))(derivative(dataset[checkup_name]["I"], 5))
        tensions = np.vectorize(lambda x: relaxed_sign(x, eps))(derivative(dataset[checkup_name]["U"], 5))
        
        # Verify that derivative arrays have the same length for consistency
        print(f"{checkup_name} {'OK' if len(currents) == len(tensions) else 'NOT OK'}")
        
        # STEP 1: Initial state classification based on derivatives and commands
        # Classify each time point based on command state and derivative signs
        for current, tension, state in zip(currents, tensions, dataset[checkup_name]["Command"]):
            if state == "Pause":
                cur = "REST"  # Battery at rest/pause
            elif state == "Charge":
                if current == 0:  # Current derivative ≈ 0 (constant current)
                    cur = "CHARGE_CC"  # Constant current charging
                elif tension == 0:  # Voltage derivative ≈ 0 (constant voltage)
                    cur = "CHARGE_CV"  # Constant voltage charging
                else:
                    cur = "CHARGE_DYNAMIC"  # Dynamic charging (both I and U changing)
            else:  # Discharging
                if current == 0:  # Current derivative ≈ 0 (constant current)
                    cur = "DISCHARGE_CC"  # Constant current discharging
                elif tension == 0:  # Voltage derivative ≈ 0 (constant voltage)
                    cur = "DISCHARGE_CV"  # Constant voltage discharging
                else:
                    cur = "DISCHARGE_DYNAMIC"  # Dynamic discharging
            
            # Add the classified state to the list
            advanced_state[checkup_name].append(cur)
        
        # STEP 2: Segment-based state smoothing
        # Initialize counter dictionary to track state occurrences within segments
        cur = {"REST": 0, "CHARGE_CC": 0, "CHARGE_CV": 0, "CHARGE_DYNAMIC": 0,
               "DISCHARGE_CC": 0, "DISCHARGE_CV": 0, "DISCHARGE_DYNAMIC": 0}
        
        # Start counting with the first state in the sequence
        cur[advanced_state[checkup_name][0]] = 1
        
        # Process each time point to detect segment boundaries and smooth within segments
        for i in range(1, len(dataset[checkup_name]["Line"])):
            # Check if we're still in the same segment (Line value hasn't changed)
            if dataset[checkup_name]["Line"][i] == dataset[checkup_name]["Line"][i - 1]:
                # Still in same segment - increment count for current state
                cur[advanced_state[checkup_name][i]] += 1
            else:
                # Segment boundary detected - Line value changed
                # STEP 2a: Find the most frequent state in the completed segment
                most_frequent_label = max(cur, key=cur.get)
                
                # STEP 2b: Retroactively assign most frequent label to entire segment
                # Start from the end of the segment and work backwards
                advanced_state[checkup_name][i - 1] = most_frequent_label
                for j in range(i - 2, -1, -1):  # Go backwards from i-2 to 0
                    # Continue assigning if still in the same segment
                    if dataset[checkup_name]["Line"][j] == dataset[checkup_name]["Line"][j + 1]:
                        advanced_state[checkup_name][j] = most_frequent_label
                    else:
                        # Reached start of segment, stop
                        break
                
                # STEP 2c: Reset counters for the new segment
                # Initialize all counts to 0, then set current state to 1
                for key in cur:
                    if key == advanced_state[checkup_name][i]:
                        cur[key] = 1  # Current state gets count of 1
                    else:
                        cur[key] = 0  # All other states get count of 0
        
        # STEP 3: Handle the final segment (no more Line changes after last point)
        i = len(dataset[checkup_name]["Line"]) - 1  # Index of last element
        most_frequent_label = max(cur, key=cur.get)  # Find most frequent in final segment
        
        # Assign most frequent label to the final segment
        advanced_state[checkup_name][i] = most_frequent_label  # Set last element
        for j in range(i - 1, -1, -1):  # Work backwards from second-to-last
            # Continue assigning if still in the same segment
            if dataset[checkup_name]["Line"][j] == dataset[checkup_name]["Line"][j + 1]:
                advanced_state[checkup_name][j] = most_frequent_label
            else:
                # Reached start of final segment, stop
                break
    
    return advanced_state