#!/usr/bin/env python3

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

prefix = sys.argv[1]
filenames = [f for f in os.listdir(".") if f.startswith(prefix) and f.endswith(".csv")]

_, ax = plt.subplots(1)

y_acc = None
for filename in filenames:
    df = pd.read_csv(filename, header=0)
    y = 100 * df["metric"].values
    if y_acc is None:
        y_acc = y
    else:
        y_acc += y
    x = np.arange(1, len(y)+1)
    ax.plot(x, y, color='blue', alpha=0.25)

y_acc /= len(filenames)
x = np.arange(1, len(y_acc)+1)
ax.plot(x, y_acc, color='blue', alpha=1.0)


# reference for comparison
# N: number of classes
# assume N one-class classifiers with FPR = 10% and FNR = 0%
# Then FP = (Total - Total/N) * FPR, the more classes, the more false positives
# The probability of not making any mistake with any classifier
# P(No-Mistake) = (1 - FP / Total) ^ N
# Even if they make a mistake, we can still pick one randomly at get it right, so:
# P(OK) = P(No-Mistake) + P(One-FP) * 1/2 + P(Two-FP) * 1/3 * ...
#       = SUM(1/(k+1) (FP/Total)^k (1 - FP/Total)^(N-k) ) for k in [0, N-1]
#         (there can be 0 mistake (k=0) but they cannot be all mistaken (k != N))

y_acc = []
FPR = 0.05
for N in x:
    total = 10 * N
    FP = (total - total / N) * FPR
    p_ok = sum([
        1/(k+1) * (FP/total)**k * (1 - FP/total)**(N-k)
        for k in range(N)
    ])
    y_acc.append(100 * p_ok)

ax.plot(x, y_acc, color='green', label="one-class FPR=5% FNR=0%")
ax.set_ylabel('accuracy (%)')
ax.set_xlabel('new classes (EMNIST letters)')
ax.set_ylim(bottom=0)
ax.set_xlim(left=1)
plt.title("Class-IL on EMNIST")
plt.legend()
plt.savefig("emnist_results.png", dpi=80, orientation='landscape')
