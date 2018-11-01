import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 21)
y = x ** 2

section = 3

if section == 1:
	fig,axes = plt.subplots(nrows=1, ncols=2)

	axes[0].plot(x, y)
	axes[1].plot(y, x)

	axes[0].set_title("First Plot")
	axes[1].set_title("Second Plot")

elif section == 2:
	fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6),dpi=100)

	# ax = fig.add_axes([0, 0, 1, 1])
	axes[0].plot(x, y, 'g-')
	axes[1].plot(y, x, 'r-')

	axes[0].set_title('Left Plot')
	axes[1].set_title('Right Plot')

elif section == 3:
	fig = plt.figure(figsize=(8, 6), dpi = 100)

	ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	ax.plot(x, x**2, label = 'Squared')
	ax.plot(x, x**3, label = 'Cubed')

	ax.legend(loc=9)

	

# plt.tight_layout()
# plt.show()
fig.savefig('my_plot.png')