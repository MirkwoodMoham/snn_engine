#pragma once

#include <utils/cuda_header.h>
#include <utils/launch_parameters.cuh>


void init_pos_gpu(
    int N, 
    int G, 
    float * N_pos,
    int * N_G,
    int G_shape_x, int G_shape_y, int G_shape_z
);