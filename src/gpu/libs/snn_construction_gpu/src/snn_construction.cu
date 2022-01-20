#include <snn_construction.cuh>


void __global__ fill_N_G_group_id_and_G_neuron_count_per_type_(
	const int N,
    const int G,
	const float* N_pos, 
    const int N_pos_n_cols,
	int* N_G,
    const int N_G_n_cols,
	const int N_G_neuron_type_col,
    const int N_G_group_id_col,

	const float G_shape_x,
	const float G_shape_y,
	const float G_shape_z,
	const float min_G_shape,

	int* G_neuron_counts  // NOLINT(readability-non-const-parameter)
)
{
	const int n = blockIdx.x * blockDim.x + threadIdx.x;  // NOLINT(bugprone-narrowing-conversions)
	
	if (n < N)
	{
		const int x = __float2int_rd((N_pos[n * N_pos_n_cols] * min_G_shape));  
		const int y = __float2int_rd((N_pos[n * N_pos_n_cols + 1] * min_G_shape));
		const int z = __float2int_rd((N_pos[n * N_pos_n_cols + 2] * min_G_shape));

		const int group = x + G_shape_x * y + G_shape_x * G_shape_y * z;
        
		const int row_idx = n * N_G_n_cols;

		// assign neuron to location-based group 
		N_G[row_idx + N_G_group_id_col] = group;
		
		// count group: row <> neuron type (1 or 2), column <> group id
		atomicAdd(&G_neuron_counts[group + G * (N_G[row_idx + N_G_neuron_type_col] - 1)], 1);	
	}
}


void fill_N_G_group_id_and_G_neuron_count_per_type(
    const int N, 
    const int G, 
    float* N_pos,
    int* N_G,
	int* G_neuron_counts,
    const int G_shape_x, const int G_shape_y, const int G_shape_z,
	const int N_pos_n_cols,
	const int N_G_n_cols,
	const int N_G_neuron_type_col,
	const int N_G_group_id_col
)
{	
	// Assign location-based group ids to neurons w.r.t. their positions.

	// N: 				number of neurons
	// G: 				number of location-based groups
	// N_pos: device 	pointer to the position array
	// N_pos_n_cols: 	number of columns of N_pos
	// N_G: 			device pointer to the neuron-group-info array
	// N_G_n_cols:	 	number of columns of N_G
	// N_G_g_id_col:	column in which to write the group-id
	// G_shape_*:		number of location-based groups along the *-axis
	// 
	//
	// Example:
	// 
	//	G = 8
	//	N_pos_n_cols = 3
	//	N_pos = [[0,0,0], 
	//			 [1,1,0]]
	//		
	//	N_G_group_id_col = 1
	//
	// 	N_G = 	[[0,0],    -> 	N_G = 	[[0,0]
	//			 [1,0]]					 [1,2]]
	//	
	//	G_neuron_counts = 	[[0,0,0],	-> 	[[1,0,0],
	//						 [0,0,0]]		 [0,0,1]]
	//	


	cudaDeviceSynchronize();
	LaunchParameters launch(N, (void *)fill_N_G_group_id_and_G_neuron_count_per_type_); 
	
	int min_G_shape = std::min(std::min(G_shape_x, G_shape_y), G_shape_z);

	fill_N_G_group_id_and_G_neuron_count_per_type_ KERNEL_ARGS2(launch.grid3, launch.block3) (
		N,
        G,
		N_pos,
        N_pos_n_cols,
		N_G,
		N_G_n_cols,
		N_G_neuron_type_col,
		N_G_group_id_col,
		// LG_neuron_counts.dp,
		static_cast<float>(G_shape_x), static_cast<float>(G_shape_y), static_cast<float>(G_shape_z),
		static_cast<float>(min_G_shape),
		G_neuron_counts
	);
	
	cudaDeviceSynchronize();

}


__global__ void fill_G_neuron_count_per_delay_and_G_synapse_count_per_delay_(
		const float fS,
		const int G,
		const int D,
		const float fD,
		const int* G_delay_distance,
		float* G_conn_probs,
		int* G_neuron_counts
		// int* G_synapse_count_per_delay,
)
{
	// connection probabilities from inh -> exc
	// connection probabilities from exc -> (inh & exc)

	const int g = threadIdx.x + blockDim.x * blockIdx.x;

	if (g < G)
	{
		{
			const int g_row = g * G;
			
			int delay = 0;
			int count_inh = 0;
			int count_exc = 0;

			const int ioffs_inh = 2 * G + g;
			const int ioffs_exc = (2 + D) * G + g;
		
			for (int h = 0; h < G; h++)
			{

				delay = G_delay_distance[g_row + h];

				count_inh = G_neuron_counts[h];
				count_exc = G_neuron_counts[h + G];

				atomicAdd(&G_neuron_counts[delay * G + ioffs_inh], count_inh);
				atomicAdd(&G_neuron_counts[delay * G + ioffs_exc], count_exc);
			}
		}
	}
}


void fill_G_neuron_count_per_delay_and_G_synapse_count_per_delay(
	const int S,
	const int G,
	const int D,
	const int* G_delay_distance,
	float* G_conn_probs,
	int* G_neuron_counts,
	int* G_synapse_count_per_delay
)
{	
	cudaDeviceSynchronize();
	LaunchParameters launch(G, (void *)fill_G_neuron_count_per_delay_and_G_synapse_count_per_delay_); 

	fill_G_neuron_count_per_delay_and_G_synapse_count_per_delay_ KERNEL_ARGS2(launch.grid3, launch.block3)(
		static_cast<float>(S),
		G,
		D,
		static_cast<float>(D),
		G_delay_distance,
		G_conn_probs,
		G_neuron_counts
		// G_synapse_count_per_delay
	);
	
	cudaDeviceSynchronize();

}
