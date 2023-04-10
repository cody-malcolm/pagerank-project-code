# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# formatting.py

import numpy as np


def vector_to_ranking(x: np.array, p: int) -> None:
    """Displays the page ranking given a rank vector x"""
    n: int = x.shape[0]

    # create a dictionary to represent the page ranks
    pages: dict = {}
    for i in range(n):
        pages[f"P({i+1})"] = x[i]

    sortedPages: list = sorted(pages.items(), key=lambda item: item[1], reverse=True)

    print("\n Ranking \n")
    for i in range(n):
        print("{0}: {1} - {2}".format(i + 1,
              sortedPages[i][0], f'{sortedPages[i][1]:.{p}f}'))

    print("\n-----------------------------------------------------------")
