# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# pagerank.py

import numpy as np
import sys
from fileReading import getHFromFile, getSettingsFromFile, getX0FromFile
from formatting import vectorToRanking


def diffIsSmaller(x, xi, res) -> bool:
    """Returns True if the sum of the differences between the entries of x and xi
    is smaller than the set residual
    """
    sum = 0
    for i in range(x.shape[0]):
        sum += abs(x[i] - xi[i])
        if sum > res:
            return False
    return True


def generateX0(n):
    """Returns a probability vector of size n where all probabilities are the same"""
    # Vector filled with 1's
    x0 = np.ones(n)
    # Normalizing to generate probability vector
    x0 = x0 / x0.sum(axis=0, keepdims=1)

    return x0


def applyIterativeMethod(H, x0, k, res):
    """Implementation of the iterative method for the PageRank Algorithm.
    Returns the ranking vector
    """
    # check stopping condition between k and res. Residual takes precendece if both are set.
    # Make sure no combination "breaks" it, should work with k, res, or k + res
    # if using k, basically the same idea as in Ex496.py
    # if using residual, similar to Ex496.py but each iteration we check to see if diff less than residual
    x = np.copy(x0)  # solution vector
    xi = np.copy(x0)  # iteration vector

    print("\n Iterative method \n")
    print("k =", k, ", res =", res)
    print("H = \n", H)
    print("x0 =", x0)

    for i in range(k):
        xi = x
        x = np.matmul(H, xi)
        if res != None:
            if diffIsSmaller(x, xi, res):
                break

    print("\n Ranking vector \n")
    print("x =", x)
    vectorToRanking(x)
    return x


def applyPowerIterativeMethod(H0, x0, k):
    """Implementation of the power iteration method for the PageRank Algorithm.
    Returns the ranking vector
    """
    H = np.copy(H0)

    print("\n Power Iteration method \n")
    print("k =", k)
    print("H = \n", H)
    print("x0 =", x0)

    for i in range(k):
        H = np.matmul(H, H0)

    x = np.matmul(H, x0)

    print("\n Ranking vector \n")
    print("x =", x)
    vectorToRanking(x)
    return x


def applyDominantEigenvectorMethod(H):
    """Implementation of the Dominant Eigenvector method for the PageRank Algorithm.
    Returns the ranking vector
    """
    print("\n Dominant Eigenvector method \n")
    print("H = \n", H)
    x = ""

    # We just need the dominant eigenvector of H, then to "normalize" it to sum to 1
    eigValues, eigVectors = np.linalg.eig(H)

    domEigenvector = []
    for entry in eigVectors:
        domEigenvector.append(round(entry[0].real, 7))

    domEigenvector = np.array(domEigenvector)

    print("\nDominant eigenvalue = %.5f" % eigValues[0].real)
    print("Dominant eigenvector = ", domEigenvector)

    # This is acting up, getting a -0 ?
    x = domEigenvector / domEigenvector.sum(axis=0, keepdims=1)

    print("\n Ranking vector \n")
    print("x = ", x)
    vectorToRanking(x)
    return x


def main():
    # get arguments from user
    # - settings file (eg. bool to apply random surfer, bool to use custom initial ranks,
    # flag to indicate which of iteration, power iteration, and/or dominant eigenvector to use,
    # int k and/or float residual, number of decimals to round to)
    # - H file: a file to read in the hyperlink matrix, I recommend .csv or .tsv
    # - x0 file: optional depending on args, a .csv or .tsv to specify initial weights

    settingsFile = sys.argv[1]
    hMatrixFile = sys.argv[2]
    if len(sys.argv) == 4:
        xVectorFile = sys.argv[3]
    else:
        xVectorFile = None

    # Set up all necessary variables and flags
    settings = getSettingsFromFile(settingsFile)
    H, n = getHFromFile(hMatrixFile, (settings["applyRandomSurfer"] == "True"))

    x0 = []
    # check settings.usingCustomInitialRanks and call getXoFromFile or generateX0 as appropriate
    if settings["usingCustomInitialRanks"] == "True" and xVectorFile != None:
        x0 = getX0FromFile(xVectorFile, n)
    else:
        x0 = generateX0(n)

    if settings["iterative"] == "True":  # if iterative flag is set
        if settings["res"] != None:
            iterationX = applyIterativeMethod(
                H, x0, int(settings["k"]), float(settings["res"])
            )
        else:
            iterationX = applyIterativeMethod(
                H, x0, int(settings["k"]), settings["res"]
            )

    if settings["power"] == "True":  # if power iterative flag is set
        powerIterationX = applyPowerIterativeMethod(H, x0, int(settings["k"]))

    if settings["eigenvector"] == "True":  # if eigenvector method flag is set
        eigenvectorX = applyDominantEigenvectorMethod(H)


main()
