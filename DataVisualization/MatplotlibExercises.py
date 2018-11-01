import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0, 100)
y = x * 2
z = x ** 2

section = 3


if section == 1:
	fig = plt.figure()
	axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	axes.plot(x, y)

	axes2 = fig.add_axes([0.2, 0.6, 0.2, 0.2])
	axes2.plot(x, y)

elif section == 2:
	fig = plt.figure()
	axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	axes2 = fig.add_axes([0.2, .4, .4, .4])

	axes1.plot(x, z)
	axes1.set_ylim([0, 10000])

	axes2.plot(x, y)
	axes2.set_xlim([20, 22])
	axes2.set_ylim([30, 50])

elif section == 3:
	fig, axes = plt.subplots(1, 2, figsize=(12, 2))

	axes[0].plot(x, y, ls="--", lw = 2, color ="blue")
	axes[1].plot(x, z, lw = 3, color="red")


plt.show()