# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# pagerank.py

import numpy as np
import sys
import matplotlib.pyplot as plt
from modules.file_reading import get_H_from_file, get_settings_from_file, get_x0_from_file
from modules.formatting import vector_to_ranking


def plot_results(results: np.array) -> None:
    """Plots the results in a scatter plot"""
    _, ax = plt.subplots()

    k: np.array = np.arange(results.shape[0])
    for i in range(results.shape[1]):
        y = results[:, i]  
        ax.scatter(k, y, label="Page " + str(i+1))
    
    plt.xlabel("Iteration")
    plt.ylabel("Page rank")
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
    plt.tight_layout()
    plt.show()
    plt.figure().clear()
    plt.close()


def generate_x0(n: int) -> np.array:
    """Returns a probability vector of size n where all probabilities are the same"""
    # Vector filled with 1's
    x0: np.array = np.ones(n)
    # Normalizing by the sum to generate probability vector
    x0 = x0 / x0.sum(axis=0, keepdims=1)

    return x0

# does not mutate H or x0
def apply_iteration(H: np.array, x0: np.array, settings: dict) -> np.array:
    """Implementation of the iterative method for the PageRank Algorithm.
    Returns the ranking vector
    """
    k: int = settings["k"]
    res: float or None = settings["res"]
    use_probability_normalization: bool = settings["apply_probability_normalization"]
    plot_requested: bool = settings["plot_results"]

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

    results: list = [x0]

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
        if plot_requested:
            results.append(np.copy(x))

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
    
    if plot_requested:
        plot_results(np.array(results))

    return x


def apply_power_iteration(H0: np.array, x0: np.array, settings: dict) -> np.array:
    """Implementation of the power iteration method for the PageRank Algorithm.
    Returns the ranking vector
    """
    k: int = settings["k"]
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

    # We just need the dominant eigenvector of H, then to apply probability normalization to it
    eigenvalues, eigenvectors = np.linalg.eig(H)

    # the rounding and +0 is to avoid "-0" in the output
    dominant_eigenvector: np.array = np.around(np.array(eigenvectors[:, 0]).real, precision + 2) + 0

    # flip signs if eigenvector has negative signs
    if (np.sum(dominant_eigenvector < 0)):
        dominant_eigenvector *= -1

    print("\nDominant eigenvalue = %.2f" % eigenvalues[0].real)
    print("Dominant eigenvector = ", dominant_eigenvector)

    return dominant_eigenvector / dominant_eigenvector.sum(axis=0, keepdims=1)


def display_results(x: np.array, p: int) -> None:
    """Displays the solution array, to precision p"""
    print("\n Ranking vector \n")
    print("x =", x)
    vector_to_ranking(x, p)


def main():
    """Main method"""
    if len(sys.argv) < 3:
        print("Incorrect usage. Expected: 'python3 pagerank.py <settings_file> <h_matrix_file> [x_vector_file]'")
        print("See README for more details.")
        exit(-1)

    # Read in the input filenames
    settings_file: str = sys.argv[1]
    h_matrix_file: str = sys.argv[2]
    if len(sys.argv) == 4:
        x_vector_file = sys.argv[3]
    else:
        x_vector_file = None

    # Set up all necessary variables and flags
    settings: dict = get_settings_from_file(settings_file)
    H, n = get_H_from_file(h_matrix_file, settings)

    p: int = int(settings["precision"])

    x0: list = []
    
    # don't bother to construct x0 if it will not be used
    if settings["iterative"] or settings["power"]:
        if x_vector_file != None:
            x0 = get_x0_from_file(x_vector_file, n)
        else:
            x0 = generate_x0(n)
        
    # set numpy to print the requested number of decimal places
    np.set_printoptions(formatter={'float': lambda x: f'{x:.{p}f}'})

    if settings["iterative"]:  # if iterative flag is set
        iteration_result: np.array = apply_iteration(H, x0, settings)
        display_results(iteration_result, p)

    if settings["power"]:  # if power iterative flag is set
        power_iteration_result: np.array = apply_power_iteration(H, x0, settings)
        display_results(power_iteration_result, p)

    if settings["eigenvector"]:  # if eigenvector method flag is set
        eigenvector_result: np.array = apply_dominant_eigenvector_method(H, p)
        display_results(eigenvector_result, p)


main()
