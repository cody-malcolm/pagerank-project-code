# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# pagerank.py

import numpy as np
import csv  

def getSettingsFromFile(file):
    # implement a dictionary, struct, or class to store the settings
    settings = {}

    # read/parse settings file and update settings appropriately, use default values where appropriate
    # If and only if neither residual or k is set, use residual = 10^-4
    # If k is not set, use k = 100
    settings['residual'] = 10 ** (-4)
    settings['k'] = 100
    settings['applyRandomSurfer'] = True

    # return the settings object
    return settings

def getHFromFile(file, applyRandomSurfer):
    # Read from file, expect a matrix that is 0's and 1's, 1s for outlinks. We will "normalize" columns to probability vectors later
    h = []
    with open(file) as infile:
        file_contents = csv.reader(infile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in file_contents: # each row is a list
            h.append(row)

    H = np.array(h) 
    
    n = H.shape[0] # determine n
    # throw exception with descriptive error if matrix is not square
    if n != H.shape[1]:
        raise Exception('Hyperlink matrix is not square')

    # if applyRandomSurfer, add 1 to every cell in matrix
    if applyRandomSurfer:
        H = H + 1
        # "normalize" each column to probability vector
        H = H/H.sum(axis=0,keepdims=1)
    else: # if !applyRandomSurfer, then it may not be a probability vector, in which case print a warning to console but allow to continue
        print("Warning: H columns are not probability vectors")

    print(H)
    print(n)
    return H, n

def getX0FromFile(file, n):
    x0 = []
    with open(file) as infile:
        file_contents = csv.reader(infile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        x0 = file_contents
    
    x0 = np.array(x0)
    # throw exception with descriptive error if length of input vector is not n
    if x0.shape != n:
        raise Exception('X_0 length is not equal to n')

    # display warning message to console if x0 is not a probability vector, and "normalize"
    # TODO

    return x0

def generateX0(n):
    x0 = ""
    # just return a n-length vector with equal weights that sum to 1
    return x0

def applyIterativeMethod(H, x0, k, res):
    # check stopping condition between k and res. Residual takes precendece if both are set. Make sure no combination "breaks" it, should work with k, res, or k + res
    # if using k, basically the same idea as in Ex496.py
    # if using residual, similar to Ex496.py but each iteration we check to see if diff less than residual
    x = "" # solution vector
    return x

def applyPowerIterativeMethod(H0, x0, k):
    H = H0 # This almost certainly requires some sort of np.copyarray type deal to work as expected
    
    for i in range(k):
        H = np.matmul(H, H0)
    
    return np.matmul(H, x0)

def applyDominantEigenvectorMethod(H):
    # Still need to figure out the way to implement this. Does numpy/scipy have something built in?
    # We just need the dominant eigenvector of H, then to "normalize" it to sum to 1
    None

def main():
    # get arguments from user
    # I recommend getting 2-3 arguments from command line, 2-3 files
    # If you prefer, you can read the filenames from stdin, advantage is then we can easily loop main with different file inputs but it is more visually messy 
    # - settings file (eg. bool to apply random surfer, bool to use custom initial ranks, flag to indicate which of iteration, power iteration, and/or dominant eigenvector to use, int k and/or float residual, number of decimals to round to)
    # - H file: a file to read in the hyperlink matrix, I recommend .csv or .tsv
    # - x0 file: optional depending on args, a .csv or .tsv to specify initial weights

    settingsFile = ""
    hMatrixFile = ""
    xVectorFile = ""

    settings = getSettingsFromFile(settingsFile)
    H, n = getHFromFile(hMatrixFile, settings['applyRandomSurfer'])

    x0 = []
    # check settings.usingCustomInitialRanks and call getXoFromFile or generateX0 as appropriate
    if settings['usingCustomInitialRanks']:
        x0 = getX0FromFile(xVectorFile, n)
    # TODO else

    #iterationX = applyIterativeMethod(H, x0, settings.k, settings.res) # if iterative flag is set

    #powerIterationX = applyPowerIterativeMethod(H, x0, settings.k) # if power iterative flag is set, note residual is meaningless here

    #eigenvectorX = applyDominantEigenvectorMethod(H) # if eigenvector method flag is set, both k and residual are meaningless here, and x0 is not even used for our purposes

    # display output including what inputs were used to generate the given output, for each method that was flagged
    # output should include the final "X" vector as well as explicitly ranking from largest to smallest

main()
