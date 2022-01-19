#pragma once

#include <utils/cuda_header.h>
#include <utils/launch_parameters.cuh>


void set_G_info(
    int N, 
    int G, 
    float* N_pos,
    int* N_G,
    int* G_neuron_counts,
    int G_shape_x, int G_shape_y, int G_shape_z,
    float N_pos_shape_x = 1, float N_pos_shape_y = 1, float N_pos_shape_z = 1,
    int N_pos_n_cols = 13,
    int N_G_n_cols = 3,
    int N_G_neuron_type_col = 0,
    int N_G_group_id_col = 1
);