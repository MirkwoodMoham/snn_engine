import numpy as np
import time

import os

print(os.getcwd())

# from build.Debug import snn_engine_gpu
from build import snn_engine_gpu

snn_engine_gpu.pyadd(1, 1, 1)

p = snn_engine_gpu.CudaGLResource()

print(p)
print(p.id)
print(p.size)
print(p.is_mapped)
p.register(2)
p.mapping()
print(p.is_mapped)


# p.age = 12
#
# size = 100000000
# arr1 = np.linspace(1.0,100.0, size)
# arr2 = np.linspace(1.0,100.0, size)
#
# runs = 10
# factor = 3.0
#
# t0 = time.time()
# for _ in range(runs):
#     snn_engine_gpu.multiply_with_scalar(arr1, factor)
# print("libs time: " + str(time.time()-t0))
# t0 = time.time()
# for _ in range(runs):
#     arr2 = arr2 * factor
# print("cpu time: " + str(time.time()-t0))
#
# print("results match: " + str(np.allclose(arr1,arr2)))
