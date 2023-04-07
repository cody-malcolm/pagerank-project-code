# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# file_reading.py

import csv
import numpy as np

def getSettingsFromFile(file):
    # Placeholder until I define settings file format/ how I want to do it
    # implement a dictionary, struct, or class to store the settings
    settings = {}

    # read/parse settings file and update settings appropriately, use default values where appropriate
    # If and only if neither residual or k is set, use residual = 10^-4
    # If k is not set, use k = 100
    settings["residual"] = 10 ** (-4)
    settings["k"] = 100
    settings["applyRandomSurfer"] = True
    settings["usingCustomInitialRanks"] = True
    settings["iterative"] = True
    settings["power"] = True
    settings["eigenvector"] = False

    # return the settings object
    return settings


def getHFromFile(file, applyRandomSurfer):
    # Read from file, expect a matrix that is 0's and 1's, 1s for outlinks. We will "normalize" columns to probability vectors later
    H = []
    with open(file) as infile:
        file_contents = csv.reader(
            infile, quoting=csv.QUOTE_NONNUMERIC
        ) 
        for line in file_contents: 
            H.append(line)

    H = np.array(H)

    n = H.shape[0]  # determine n
    # throw exception with descriptive error if matrix is not square
    if n != H.shape[1]:
        raise Exception("Hyperlink matrix is not square")

    # if applyRandomSurfer, add 1 to every cell in matrix
    if applyRandomSurfer:
        H = H + 1
        # "normalize" each column to probability vector
        H = H / H.sum(axis=0, keepdims=1)
    else:  # if !applyRandomSurfer, then it may not be a probability vector, in which case print a warning to console but allow to continue
        print("Warning: H columns are not probability vectors")

    return H, n


def getX0FromFile(file, n):
    x0 = []
    with open(file) as infile:
        file_contents = csv.reader(
            infile, quoting=csv.QUOTE_NONNUMERIC
        )
        for line in file_contents: 
            x0 = line

    x0 = np.array(x0)

    # throw exception with descriptive error if length of input vector is not n
    if x0.shape[0] != n:
        raise Exception("x_0 length is not equal to n")

    # display warning message to console if x0 is not a probability vector, and "normalize"
    if np.sum(x0) != 1:
        print('Warning: x_0 is not a probability vector, normalizing...')
        x0 = x0 / x0.sum(axis=0, keepdims=1)

    return x0