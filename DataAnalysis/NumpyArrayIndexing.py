import numpy as np

arr = np.arange(0, 11)
print(arr)
print(arr[8])
print(arr[1:5])
print(arr[0:5])
print(arr[6:])
arr[7:] = 100
print(arr)
arr = np.arange(0, 11)
slice_arr = arr[0:6]
print(slice_arr)
slice_arr[:] = 100
print(slice_arr)
print(arr)

arr = np.arange(0, 11)
print(arr)
print(slice_arr)
slice_arr = arr[0:6].copy()
print(slice_arr)
slice_arr[:] = 100
print(slice_arr)
print(arr)


arr2 = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
print(arr2)

print(arr2[1])
print(arr2[2][2])
print(arr2[1, 2])
print(arr2[1, 0:2])
print(arr2[1,:])
print(arr2[1,:] == 25)
print(arr2[1, :] * 4)

print(arr[arr > 5])


arr2 = np.arange(50).reshape(5, 10)
print(arr2)
print(arr2[2, 3:6])