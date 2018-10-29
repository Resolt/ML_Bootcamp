import numpy as np

# LISTS AND ARRAYS
lst = [1, 2, 3, 4, 5]
arr = np.array(lst)
print(arr)

lst2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
arr2 = np.array(lst2)
print(arr2)

# RANGE
print(np.arange(5))

# ZEROS AND ONES
print(np.zeros(10))
print(np.zeros((2, 2)))
print(np.ones((4, 4, 4, 4)))

# RANDOM
print(np.random.rand(5))
print(np.random.randint(0, 100, 1))

# RESHAPE
arr3 = np.arange(10)
print(arr3)
print(arr3.reshape(5, 2))

# METHODS
ranarr = np.random.randint(5, 55, 15)
print(ranarr.max())
print(ranarr.argmax())
if ranarr.max() == ranarr[ranarr.argmax()]:
	print('Yes!')

print(ranarr.shape)
print(ranarr.dtype)

# IMPORT
from numpy.random import randint as rand

ran = rand(0, 100, 5)
print(ran)