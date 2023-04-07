# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# file_reading.py

import csv
import numpy as np
import re


def columnSum(n, matrix):
    """Returns the sum of all entries in column n in matrix"""
    sum = 0
    for row in range(matrix.shape[1]):
        sum += matrix[row][n]
    return sum


# res should be a value that is either in 0.00001 or 10^-4 or 2x10^-4 format
def getResidual(string):
    format1 = "[0-9][.][0-9]+"
    format2 = "^[10\^]\-?[0-9]+"
    format3 = "[0-9]+[x][1][0]\^\-?[0-9]+"

    if re.search(format1, string):
        return float(string)
    elif re.search(format2, string):
        return 10 ** (int(string[3::]))
    elif re.search(format3, string):
        xPos = string.find("x")
        powerPos = string.find("^") + 1
        return int(string[0:xPos]) * (10 ** int(string[powerPos::]))
    else:
        return None


def getSettingsFromFile(file):
    """Reads the settings file and returns a dictionary with the properties and flags"""
    separator = "="
    settings = {}

    # read/parse settings file and update settings appropriately, use default values where appropriate
    with open(file) as infile:
        for line in infile:
            if separator in line:
                key, value = line.split(separator, 1)
                settings[key] = value.strip()

    # If k is not set, use k = 100
    if "k" not in settings:
        settings["k"] = 100
        # If and only if neither residual or k is set, use residual = 10^-4
        if "res" not in settings:
            settings["res"] = 0.0001
    elif "res" not in settings:
        settings["res"] = None

    if "res" in settings:
        settings["res"] = getResidual(settings["res"])

    if "precision" not in settings:
        settings["precision"] = 2

    # return the settings object
    return settings


def getHFromFile(file, settings):
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

    # if applyRandomSurfer, add 1 to every cell in matrix
    if settings["applyRandomSurfer"] == "True":
        H = H + 1
        # 'normalize' each column to probability vector

    H = H / H.sum(axis=0, keepdims=1)
    if settings["applyRandomSurfer"] != "True":
        # Check if there is a 0 column and print the message only in that case
        for i in range(H.shape[0]):
            if columnSum(i, H) == 0:
                print("Warning: H columns are not probability vectors")
                break

    return H, n


def getX0FromFile(file, n):
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

    # display warning message to console if x0 is not a probability vector, and 'normalize'
    if np.sum(x0) != 1:
        print("Warning: x_0 is not a probability vector, normalizing...")
        x0 = x0 / x0.sum(axis=0, keepdims=1)

    return x0
