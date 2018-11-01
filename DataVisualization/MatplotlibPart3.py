import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 21)
y = x ** 2

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

ax.plot(x, y, color='purple', lw=3, alpha=0.6, ls='-', marker='o', ms=10)

ax.set_xlim([0, 1])
ax.set_ylim([0,2])

fig.savefig('my_plot.png')