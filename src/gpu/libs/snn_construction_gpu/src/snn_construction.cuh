#pragma once

#include <utils/cuda_header.h>
#include <utils/launch_parameters.cuh>


void fill_N_G_group_id_and_G_neuron_count_per_type(
    int N, 
    int G, 
    float* N_pos,
    int* N_G,
    int* G_neuron_counts,
    int G_shape_x, int G_shape_y, int G_shape_z,
    int N_pos_n_cols = 13,
    int N_G_n_cols = 3,
    int N_G_neuron_type_col = 0,
    int N_G_group_id_col = 1
);


void fill_G_neuron_count_per_delay_and_G_synapse_count_per_delay(
	const int S,
	const int G,
	const int D,
	const int* G_delay_distance,
	float* G_conn_probs,
	int* G_neuron_counts,
	int* G_synapse_count_per_delay
);