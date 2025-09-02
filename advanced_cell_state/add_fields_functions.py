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