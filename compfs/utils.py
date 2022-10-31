# third party
import numpy as np


def is_array_in_list(arr, arr_list):
    """Checks if a trial array is in a list of arrays."""
    for element in arr_list:
        if np.array_equal(element, arr):
            return True
    return False
