from gpu.libs.snn_construction_gpu.build import (
    snn_construction_gpu,
    snn_simulation_gpu
)

from .kernel_launch_parameter import DeviceProperties

from .gpu_arrays import (
    GPUArrayCollection,
    GPUArrayConfig,
    RegisteredGPUArray,
    RegisteredVBO,
)
# print_random_numbers
# def func2():
#     print(10)
