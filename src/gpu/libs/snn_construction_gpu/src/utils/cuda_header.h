#pragma once

#include <cuda/helper_cuda.h>

#include <cuda_runtime.h>

#include <cuda_gl_interop.h>

#include <curand.h>
#include <curand_kernel.h>

#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#ifdef __INTELLISENSE__

#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)

#else

#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>

#endif

#include <utils/launch_parameters.cuh>
#include <utils/curand_states.cuh>