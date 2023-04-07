# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# formatting.py

import numpy as np


def vectorToRanking(x, p):
    """Displays the page ranking given a rank vector x"""
    n = x.shape[0]
    pages = {}
    for i in range(n):
        if x[i] < 0.00001:
            pages[f"P({i})"] = 0.0
        else:
            pages[f"P({i})"] = round(x[i], p)

    sortedPages = sorted(pages.items(), key=lambda item: item[1])
    print("\n Ranking \n")
    for i in range(n - 1, -1, -1):
        print(sortedPages[i])

    print("\n-----------------------------------------------------------")


def printVector(x):
    print(x)
