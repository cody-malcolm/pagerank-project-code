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

# does not mutate H or x0
def apply_iteration(H: np.array , x0: np.array, settings: dict) -> np.array:
    """Implementation of the iterative method for the PageRank Algorithm.
    Returns the ranking vector
    """
    k: int = settings["k"]
    res: float or None = settings["res"]
    use_probability_normalization: bool = settings["apply_probability_normalization"]

    x: np.array = np.copy(x0)  # solution vector
    xi: np.array = np.copy(x0)  # iteration vector

    print("\n Iterative method \n")
    # print stop criteria
    if (res == None):
        print("k =", k)
    else:
        print("res =", res)

    print("H = \n", H)
    print("x0 =", x0)

    # iterate Hx_i for residual < res, or for k iterations if res is not set
    stop: bool = False # stop flag
    i: int = 0 # iteration
    prev: float = float('inf')

    while not stop:
        # perform next iteration
        xi = x
        x = np.matmul(H, xi)
        if use_probability_normalization:
            x = x / np.sum(x)
        i += 1

        if res == None: # ie. stop after k iterations
            stop = i == k
        else:
            sum: float = np.sum(np.abs(x - xi)) # calculate residual
            stop = sum < res
            if stop:
                print("\nConverged normally after", i, "iterations.")
            # due to float point imprecision, residual method may not converge properly if residual is too precise
            elif (i % 10 == 0):
                stop = prev < sum  # exit if residual has grown over 10 iterations
                if stop:
                    print("\nExiting after", i, "iterations due to floating point imprecision.")
                    print("Residual is approximately around", prev, "to", sum)
                prev = sum
    return x


def apply_power_iteration(H0: np.array, x0: np.array, settings: dict) -> np.array:
    """Implementation of the power iteration method for the PageRank Algorithm.
    Returns the ranking vector
    """
    k = settings["k"]
    use_probability_normalization: bool = settings["apply_probability_normalization"]

    H: np.array = np.copy(H0)

    print("\n Power Iteration method \n")
    print("k =", k)
    print("H = \n", H)
    print("x0 =", x0)

    for _ in range(k):
        H = np.matmul(H, H0)
        if use_probability_normalization:
            for i in range(H.shape[0]):
                sum: float = np.sum(H[:, i])
                if sum != 0:
                    H[:, i] = H[:, i] / sum

    print("H^k = \n", H)

    return np.matmul(H, x0)


def apply_dominant_eigenvector_method(H: np.array, precision: int) -> np.array:
    """Implementation of the Dominant Eigenvector method for the PageRank Algorithm.
    Returns the ranking vector
    """
    print("\n Dominant Eigenvector method \n")
    print("H = \n", H)
    x = ""

    # We just need the dominant eigenvector of H, then to "normalize" it to sum to 1
    eigenvalues, eigenvectors = np.linalg.eig(H)

    # the rounding and +0 is to avoid "-0" in the output
    dominant_eigenvector: np.array = np.around(np.array(eigenvectors[:, 0]).real, precision + 2) + 0
    
    # flip signs if eigenvector has negative signs
    if (np.sum(dominant_eigenvector < 0)):
        dominant_eigenvector *= -1

    print("\nDominant eigenvalue = %.2f" % eigenvalues[0].real)
    print("Dominant eigenvector = ", dominant_eigenvector)

    return dominant_eigenvector / dominant_eigenvector.sum(axis=0, keepdims=1)


def display_results(x: np.array, p: int):
    print("\n Ranking vector \n")
    print("x =", x)
    vector_to_ranking(x, p)

def main():
    # get arguments from user
    # - settings file (eg. bool to apply random surfer, bool to use custom initial ranks,
    # flag to indicate which of iteration, power iteration, and/or dominant eigenvector to use,
    # int k and/or float residual, number of decimals to round to)
    # - H file: a file to read in the hyperlink matrix, I recommend .csv or .tsv
    # - x0 file: optional depending on args, a .csv or .tsv to specify initial weights

    if len(sys.argv) < 3:
        print("Incorrect usage. Expected: 'python3 pagerank.py <settings_file> <h_matrix_file> [x_vector_file]'")
        print("See README for more details.")
        exit(-1)

    settings_file: str = sys.argv[1]
    h_matrix_file: str = sys.argv[2]
    if len(sys.argv) == 4:
        x_vector_file: str = sys.argv[3]
    else:
        x_vector_file: str = None

    # Set up all necessary variables and flags
    settings = get_settings_from_file(settings_file)
    H, n = get_H_from_file(h_matrix_file, settings)

    p = int(settings["precision"])

    np.set_printoptions(formatter={'float': lambda x: f'{x:.{p}f}'})

    x0 = []
    
    if x_vector_file != None:
        x0 = get_x0_from_file(x_vector_file, n)
    else:
        x0 = generate_x0(n)

    # Format matrixes to specified precision
    H = np.matrix.round(H, p)
    x0 = np.matrix.round(x0, p)

    if settings["iterative"]:  # if iterative flag is set
        if settings["res"] != None:
            iteration_result = apply_iteration(H, x0, settings)
        else:
            iteration_result = apply_iteration(H, x0, settings)
        display_results(iteration_result, p)

    if settings["power"]:  # if power iterative flag is set
        power_iteration_result = apply_power_iteration(H, x0, settings)
        display_results(power_iteration_result, p)

    if settings["eigenvector"]:  # if eigenvector method flag is set
        eigenvector_result = apply_dominant_eigenvector_method(H, p)
        display_results(eigenvector_result, p)


main()
