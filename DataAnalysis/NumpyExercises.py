import numpy as np

print(np.zeros(10))

print(np.ones(10))

arr = np.arange(10)
arr[:] = 5
print(arr)

print(np.arange(10, 51))

print(np.arange(10, 51, 2))

print(np.arange(0, 9).reshape(3, 3))

print(np.identity(3))

print(np.random.rand(1))

print(np.random.uniform(-2, 2, 25).reshape(5, 5))

print(np.arange(0.01, 1.01, 0.01).reshape(10, 10))

print(np.linspace(0, 1, 20).reshape(4, 5))

mat = np.arange(1, 26).reshape(5, 5)
print(mat)

# print(np.array((np.arange(12, 16), np.arange(17, 21), np.arange(22, 26))))
print(mat[2:,1:])

print(mat[3, 4])

# print(np.array((2, 7, 12)).reshape(3, 1))
print(mat[:3, 1:2])

print(mat[4,:])

print(mat[3:,:])

print(mat.sum())

print(mat.std())

print(np.array([mat[:, v].sum() for v in np.arange(0, np.shape(mat)[1])]))
print(mat.sum(axis = 0))

