import numpy as np

input_arr = np.full((7, 7), 1)

# Creating the array
input_arr[5:7, :] = 0
input_arr[4, 2:7] = 0
input_arr[3, 4:7] = 0
input_arr[2, 6] = 0

# Creating the kernel array
kernel_arr = np.full((3,3), -1)
kernel_arr[1, 1] = 8

# Element wise multiplication a * b
value = input_arr[0:3, 0:3] * kernel_arr
value = sum(sum(value))

# Creating Result array
result_arr = np.full((5,5), 0)
result_arr[0, 0] = value

# Convolve step
def convolve(data, kernel, r, c):
    field = data[r-1:r+2, c-1:c+2]
    return (field * kernel).sum()

# Loop for whole
n = len(input_arr) - 2
solution = np.full((n, n), 0)
for i in range(n):
    for k in range(n):
        solution[i, k] = convolve(input_arr, kernel_arr, i+1, k+1)

print(solution)