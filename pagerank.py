# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# pagerank.py

import numpy as np
import sys
from modules.file_reading import get_H_from_file, get_settings_from_file, get_x0_from_file
from modules.formatting import vector_to_ranking


def generate_x0(n: int) -> np.array:
    """Returns a probability vector of size n where all probabilities are the same"""
    # Vector filled with 1's
    x0 = np.ones(n)
    # Normalizing by the sum to generate probability vector
    x0 = x0 / x0.sum(axis=0, keepdims=1)

    return x0

# TODO: verify no combination "breaks" it, should work with k, res, or k + res
# does not mutate H or x0
def apply_iterative_method(H: np.array , x0: np.array, k: int, res: float, p: int) -> np.array:
    """Implementation of the iterative method for the PageRank Algorithm.
    Returns the ranking vector
    """
    x = np.copy(x0)  # solution vector
    xi = np.copy(x0)  # iteration vector

    print("\n Iterative method \n")
    print("k =", k, ", res =", res)
    print("H = \n", H)
    print("x0 =", x0)
    print("")

    # iterate Hx_i for residual < res, or for k iterations if res is not set
    stop = False
    i = 0
    prev = float('inf')
    while not stop:
        xi = x
        x = np.matmul(H, xi)
        i += 1
        if res == None:
            stop = i == k
        else:
            sum = np.sum(np.abs(x - xi))
            stop = sum < res
            if stop:
                print("Converged normally after", i, "iterations.")
            # due to float point imprecision, residual method may not converge properly
            if (i % 10 == 0):
                stop = prev < sum  # exit if residual has grown over 10 iterations
                if stop:
                    print("Exiting after", i, "iterations due to floating point imprecision.")
                    print("Residual is approximately around", prev, "to", sum)
                prev = sum

    x = np.matrix.round(x, p)

    print("\n Ranking vector \n")
    #print(f'x = {x:.{p}f}') 
    print("x =", x)
    vector_to_ranking(x, p)
    return x


def applyPowerIterativeMethod(H0, x0, k, p):
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
    x = np.matrix.round(x, p)

    print("\n Ranking vector \n")
    print("x =", x)
    vector_to_ranking(x, p)
    return x


def applyDominantEigenvectorMethod(H, p):
    """Implementation of the Dominant Eigenvector method for the PageRank Algorithm.
    Returns the ranking vector
    """
    H = np.matrix.round(H, p)
    print("\n Dominant Eigenvector method \n")
    print("H = \n", H)
    x = ""

    # We just need the dominant eigenvector of H, then to "normalize" it to sum to 1
    eigValues, eigVectors = np.linalg.eig(H)

    domEigenvector = []
    for entry in eigVectors:
        domEigenvector.append(round(entry[0].real, 7))

    domEigenvector = np.array(domEigenvector)
    domEigenvector = np.matrix.round(domEigenvector, p)

    print("\nDominant eigenvalue = %.2f" % eigValues[0].real)
    print("Dominant eigenvector = ", domEigenvector)

    # This is acting up, getting a -0 ?
    x = domEigenvector / domEigenvector.sum(axis=0, keepdims=1)
    x = np.matrix.round(x, p)

    print("\n Ranking vector \n")
    print("x = ", x)
    vector_to_ranking(x, p)
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
    settings = get_settings_from_file(settingsFile)
    H, n = get_H_from_file(hMatrixFile, settings)

    p = int(settings["precision"])

    np.set_printoptions(formatter={'float': lambda x: f'{x:.{p}f}'})

    x0 = []
    # check settings.usingCustomInitialRanks and call getXoFromFile or generateX0 as appropriate
    if settings["usingCustomInitialRanks"] == "True" and xVectorFile != None:
        x0 = get_x0_from_file(xVectorFile, n)
    else:
        x0 = generate_x0(n)

    # Format matrixes to specified precision
    H = np.matrix.round(H, p)
    x0 = np.matrix.round(x0, p)

    if settings["iterative"] == "True":  # if iterative flag is set
        if settings["res"] != None:
            iterationX = apply_iterative_method(
                H, x0, int(settings["k"]), float(settings["res"]), p
            )
        else:
            iterationX = apply_iterative_method(
                H, x0, int(settings["k"]), settings["res"], p
            )

    if settings["power"] == "True":  # if power iterative flag is set
        powerIterationX = applyPowerIterativeMethod(H, x0, int(settings["k"]), p)

    if settings["eigenvector"] == "True":  # if eigenvector method flag is set
        eigenvectorX = applyDominantEigenvectorMethod(H, p)


main()
