import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 21)
y = x ** 2

# plt.plot(x, y)
# plt.xlabel('X label')
# plt.ylabel('Y label')
# plt.title('title')

# plt.subplot(1, 2, 1)
# plt.plot(x, y, 'r-')

# plt.subplot(1, 2, 2)
# plt.plot(y, x, 'b-')

# plt.show()

# OBJECT ORIENTED METHOD

fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])

axes1.plot(x, y)
axes2.plot(y, x)

axes1.set_title('Large')
axes2.set_title('Small')

# axes.plot(x, y)
plt.show()