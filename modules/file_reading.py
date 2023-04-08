# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# file_reading.py

import csv
import numpy as np
import re


def column_sum(n, matrix):
    """Returns the sum of all entries in column n in matrix"""
    sum = 0
    for row in range(matrix.shape[1]):
        sum += matrix[row][n]
    return sum

def col_is_zero(col: np.array):
    return not np.any(col) # numpy hack



# res should be a value that is either in 0.00001 or 10^-4 or 2x10^-4 format
def get_residual(string: str) -> float or None:
    format1: str = "[0-9][.][0-9]+"
    format2: str = "^[10\^]\-?[0-9]+"
    format3: str = "[0-9]+[x][1][0]\^\-?[0-9]+"

    if re.search(format1, string):
        return float(string)
    elif re.search(format2, string):
        return 10 ** (int(string[3::]))
    elif re.search(format3, string):
        x_pos = string.find("x")
        power_pos = string.find("^") + 1
        return int(string[0:x_pos]) * (10 ** int(string[power_pos::]))
    else:
        return None
    

def ensure_bool_set(settings: dict, key: str, default: bool):
    if key not in settings:
        settings[key] = default
    else:
        # convert to boolean
        settings[key] = settings[key] == "True"


def get_settings_from_file(file):
    """Reads the settings file and returns a dictionary with the properties and flags"""
    separator: str = "="
    settings: dict = {}

    # read/parse settings file and update settings appropriately, use default values where appropriate
    with open(file) as infile:
        for line in infile:
            if separator in line:
                key, value = line.split(separator, 1)
                settings[key] = value.strip()

    if "res" in settings:
        settings["res"] = get_residual(settings["res"])

    # If k is not set, use k = 100
    if "k" not in settings:
        settings["k"] = 100
        # If and only if neither residual or k is set, use residual = 10^-4
        if "res" not in settings:
            settings["res"] = 0.0001
    else:
        settings["k"] = int(settings["k"])
        if "res" not in settings:
            settings["res"] = None

    # default rounding to 2 decimal places
    if "precision" not in settings:
        settings["precision"] = 2

    ensure_bool_set(settings, "apply_random_surfer", False)
    ensure_bool_set(settings, "apply_probability_normalization", True)
    ensure_bool_set(settings, "iterative", True)
    ensure_bool_set(settings, "power", False)
    ensure_bool_set(settings, "eigenvector", True)

    # return the settings object
    return settings


def get_H_from_file(file, settings):
    """Reads the Hyperlink matrix file and returns a numpy 2D array"""
    # Read from file, expect a matrix that is 0's and 1's, 1s for outlinks.
    # We will 'normalize' columns to probability vectors later
    H = []
    with open(file) as infile:
        fileContents = csv.reader(infile, quoting=csv.QUOTE_NONNUMERIC)
        for line in fileContents:
            H.append(line)

    H = np.array(H)

    n = H.shape[0]  # determine n
    # throw exception with descriptive error if matrix is not square
    if n != H.shape[1]:
        raise Exception("Hyperlink matrix is not square")

    # if apply_random_surfer, add 1 to every cell in matrix
    if settings["apply_random_surfer"]:
        H = H + 1
    
    # apply probability normalization to each column
    

    # if not using random surfer, print warning if matrix is not stochastic (has a 0 column)
    if settings["apply_random_surfer"]:
        H = H / H.sum(axis=0, keepdims=1)
    else:
        printed_warning: bool = False # so we only print a warning once
        for i in range(n):
            sum: float = np.sum(H[:, i])
            if sum == 0:
                if not printed_warning:
                    print("\nWarning: H columns are not probability vectors")
                    print("The eigenvector method relies on matrix being stochastic, and will produce incorrect results.")
                    print("Other methods will show convergence to the zero vector.")
                    printed_warning = True
            else:
                H[:, i] = H[:, i] / sum

    return H, n


def get_x0_from_file(file: str, n: int):
    """Reads the initial probability vector file and returns a numpy vector"""
    x0 = []
    with open(file) as infile:
        fileContents = csv.reader(infile, quoting=csv.QUOTE_NONNUMERIC)
        for line in fileContents:
            x0 = line

    x0 = np.array(x0)

    # throw exception with descriptive error if length of input vector is not n
    if x0.shape[0] != n:
        raise Exception("x_0 length is not equal to n")

    # display warning message to console if x0 is not a probability vector, and perform probability normalization
    if np.sum(x0) != 1:
        print("Warning: x_0 is not a probability vector, normalizing...")
        x0 = x0 / x0.sum(axis=0, keepdims=1)

    return x0
