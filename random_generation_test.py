from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_uniform_float32)
import numpy as np
import cupy as cp
import time

@cuda.jit
def random_3d(arr, rng_states):
    # Per-dimension thread indices and strides
    startx, starty, startz = cuda.grid(3)
    stridex, stridey, stridez = cuda.gridsize(3)

    # Linearized thread index
    tid = (startz * stridey * stridex) + (starty * stridex) + startx

    # Use strided loops over the array to assign a random value to each entry
    for i in range(startz, arr.shape[0], stridez):
        for j in range(starty, arr.shape[1], stridey):
            for k in range(startx, arr.shape[2], stridex):
                arr[i, j, k] = stridey

@cuda.jit
def random_1d(arr, rng_states):
    # Per-dimension thread indices and strides
    x = cuda.grid(1) # current thread id
    stridex = cuda.gridsize(1) # total number of threads (bx * gx in this case)

    for k in range(x, arr.shape[0], stridex):
        arr[k] = xoroshiro128p_uniform_float32(rng_states, x)


X = 1000 # random arr length
# Block and grid dimensions
bx = 2
gx = 4
# Total number of threads
nthreads = bx * gx
with cuda.gpus[1]:
    # Initialize a state for each thread
    rng_states = create_xoroshiro128p_states(nthreads, seed=1)
    # Generate random numbers
    arr = np.zeros(X, dtype=np.float32)
    arr = cuda.to_device(arr)
    random_1d[(gx), (bx)](arr, rng_states)
print(len(set(list(arr.copy_to_host()))))
print(arr.copy_to_host())

with cuda.gpus[1]:
    # Initialize a state for each thread
    rng_states = create_xoroshiro128p_states(nthreads, seed=1)
    # Generate random numbers
    arr1 = np.zeros(X, dtype=np.float32)
    arr1 = cuda.to_device(arr1)
    random_1d[(gx), (bx)](arr1, rng_states)
print((arr1.copy_to_host() == arr.copy_to_host()).prod())



@cuda.jit
def random_1d_modulo(arr, rng_states, numstates):
    # Per-dimension thread indices and strides
    x = cuda.grid(1) # current thread id
    stridex = cuda.gridsize(1) # total number of threads (bx * gx in this case)
    arr[x] = xoroshiro128p_uniform_float32(rng_states, x%numstates)


X = 1000 # random arr length
# Block and grid dimensions
bx = 32
gx = 1000 // bx + 1
# Total number of threads
nthreads = 8
with cuda.gpus[1]:
    # Initialize a state for each thread
    rng_states = create_xoroshiro128p_states(nthreads, seed=1)
    # Generate random numbers
    arr = np.zeros(X, dtype=np.float32)
    arr = cuda.to_device(arr)
    random_1d_modulo[(gx), (bx)](arr, rng_states, nthreads)
print(len(set(list(arr.copy_to_host()))))
print(arr.copy_to_host())

# Array dimensions
X, Y, Z = 701, 900, 719

# Block and grid dimensions
bx, by, bz = 3, 8,  4
gx, gy, gz = 6, 16, 12

# Total number of threads
nthreads = bx * by * bz * gx * gy * gz

with cuda.gpus[1]:
    # Initialize a state for each thread
    rng_states = create_xoroshiro128p_states(nthreads, seed=1)
    
    # Generate random numbers
    arr = cuda.device_array((X, Y, Z), dtype=np.float32)
    random_3d[(gx, gy, gz), (bx, by, bz)](arr, rng_states)

print(arr.copy_to_host())

def free_arr(arr):
    out_host = arr.get()
    del arr
    cp.get_default_memory_pool().free_all_blocks()
    return out_host

with cp.cuda.Device(1):
    a = cp.zeros(10**9, dtype=np.uint8)
print('gpu loaded!')
time.sleep(10)
free_arr(a)
print('gpu released')
time.sleep(10)



"""padded_size = 23
    itvsize = 23
    with cuda.gpus[1]:
        # get reshuffle index
        threadsperblock = 32
        blockspergrid = (PIN_SAMPLE * padded_size + threadsperblock - 1) // threadsperblock
        reshuidx = np.zeros(PIN_SAMPLE * padded_size, dtype=np.uint8)
        rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=0)
        convert2seed[blockspergrid, threadsperblock](rng_states, 0, 1, PIN_SAMPLE*padded_size, reshuidx, itvsize, padded_size//itvsize, padded_size if args.fixeditv and args.fixeditvmode==1 else -1)"""