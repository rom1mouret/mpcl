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
    ax.plot(x, y, color='red', alpha=0.25)

y_acc /= len(filenames)
x = np.arange(1, len(y_acc)+1)
ax.plot(x, y_acc, color='red', alpha=1.0)

ax.set_ylabel('accuracy (%)')
ax.set_xlabel('permutation tasks')
ax.set_ylim(bottom=0)
ax.set_xlim(left=1)
plt.title("Domain-IL on Permuted MNIST")
plt.savefig("mnist_results.png", dpi=80, orientation='landscape')
