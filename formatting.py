# Avril Lopez & Cody Malcolm
# April 6th, 2023
# MATH 4020U Project Code

# formatting.py

import numpy as np

def vectorToRanking(x):
    n = x.shape[0]
    pages = {}
    for i in range(0, n):
        pages[f'P({i})'] = x[i]

    sorted_pages = sorted(pages.items(), key=lambda item: item[1])
    print('\n Ranking \n')
    for i in range(n-1, -1, -1):
        print(sorted_pages[i])