# Avril Lopez & Cody Malcolm
# MATH 4020U scratch code

# This file includes the scratch code used in preparation for the presentation on March 30th
# It is included for "show your work" purposes only and is not used or required by the code submission

import numpy as np

H = np.array([[0, 0, 1/3, 0, 0, 0], [1/2, 0, 1/3, 0, 0, 0], [1/2, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1/2, 1], [0, 0, 1/3, 1/2, 0, 0], [0, 0, 0, 1/2, 1/2, 0]])

print(H)

x0 = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

print(x0)

for i in range(64):
    x = np.matmul(H, x0)
    x0 = x
    if i == 0 or i == 1 or i == 2 or i == 3 or i == 63:
        print("Iteration", i,"\n", x)


deadH = np.array([[0, 0, 1/3, 0, 0, 0], [1/2, 0, 1/3, 0, 0, 0], [1/2, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1/2, 0], [0, 0, 1/3, 1/2, 0, 0], [0, 0, 0, 1/2, 1/2, 0]])

x0 = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

for i in range(64):
    x = np.matmul(deadH, x0)
    x0 = x
    if i == 0 or i == 1 or i == 3 or i == 2 or i == 15 or i == 63:
        print("Iteration", i, "\n", x)


spiderH = np.array([[0, 0, 1/3, 0, 0, 0], [1/2, 0, 1/3, 0, 0, 0], [1/2, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1/2, 0], [0, 0, 1/3, 1/2, 0, 0], [0, 0, 0, 1/2, 1/2, 1]])

x0 = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

for i in range(64):
    x = np.matmul(spiderH, x0)
    x0 = x
    if i == 0 or i == 1 or i == 3 or i == 2 or i == 15 or i == 31 or i == 63:
        print("Iteration", i, "\n", x)



randomDeadH = np.array([[1/8, 1/7, 2/9, 1/8, 1/8, 1/6], [1/4, 1/7, 2/9, 1/8, 1/8, 1/6], [1/4, 2/7, 1/9, 1/8, 1/8, 1/6],
                        [1/8, 1/7, 1/9, 1/8, 1/4, 1/6], [1/8, 1/7, 2/9, 1/4, 1/8, 1/6], [1/8, 1/7, 1/9, 1/4, 1/4, 1/6]])

x0 = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

for i in range(256):
    x = np.matmul(randomDeadH, x0)
    x0 = x
    #if i == 0 or i == 1 or i == 3 or i == 7 or i == 15 or i == 31 or i == 63:
    if i == 255:
        print("Iteration", i, "\n", x)

sum = 245/279 + 245/248 + 63/62 + 8/9 + 92/93 + 1
v = np.array([(245/279)/sum, (245/248)/sum, (63/62)/sum, (8/9)/sum, (92/93)/sum, 1/sum])
print(v)

randomSpiderH = np.array([[1/8, 1/7, 2/9, 1/8, 1/8, 1/7], [1/4, 1/7, 2/9, 1/8, 1/8, 1/7], [1/4, 2/7, 1/9, 1/8, 1/8, 1/7],
                        [1/8, 1/7, 1/9, 1/8, 1/4, 1/7], [1/8, 1/7, 2/9, 1/4, 1/8, 1/7], [1/8, 1/7, 1/9, 1/4, 1/4, 2/7]])

x0 = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

for i in range(256):
    x = np.matmul(randomSpiderH, x0)
    x0 = x
    #if i == 0 or i == 1 or i == 3 or i == 7 or i == 15 or i == 31 or i == 63:
    if i == 255:
        print("Iteration", i, "\n", x)
