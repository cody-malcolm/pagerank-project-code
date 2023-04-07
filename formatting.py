# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# formatting.py

import numpy as np

def vectorToRanking(x):
    n = x.shape[0]
    pages = {}
    for i in range(n):
        pages[f'P({i})'] = x[i]

    sortedPages = sorted(pages.items(), key=lambda item: item[1])
    print('\n Ranking \n')
    for i in range(n-1, -1, -1):
        print(sortedPages[i])

    print('\n-----------------------------------------------------------')