import cython
import numpy as np

cdef double my_sum(double[::1] arr):
    cdef:
        double s = 0.0
        int i = 0
    for i in range(arr.shape[0]):
        s += arr[i]
    return s

@cython.boundscheck(False)
def mean3filter3(double[::1] arr):
    cdef double[::1] arr_out = np.empty_like(arr)
    cdef int i
    for i in range(1, arr.shape[0]-1):
        arr_out[i] = my_sum(arr[i-1:i+1]) / 3
    arr_out[0] = (arr[0] + arr[1]) / 2
    arr_out[-1] = (arr[-1] + arr[-2]) / 2
    return np.asarray(arr_out)
