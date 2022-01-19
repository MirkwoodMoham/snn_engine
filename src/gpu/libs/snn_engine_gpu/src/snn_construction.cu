#include <snn_construction.cuh>


void __global__ set_G_info_(
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
	
	const float N_pos_shape_x,
	const float N_pos_shape_y,
	const float N_pos_shape_z,

	int* G_neuron_counts  // NOLINT(readability-non-const-parameter)
)
{
	const int n = blockIdx.x * blockDim.x + threadIdx.x;  // NOLINT(bugprone-narrowing-conversions)
	
	if (n < N)
	{
		const int x = __float2int_rd((N_pos[n * N_pos_n_cols] * G_shape_x) / N_pos_shape_x);  
		const int y = __float2int_rd((N_pos[n * N_pos_n_cols + 1] * G_shape_y) / N_pos_shape_y);
		const int z = __float2int_rd((N_pos[n * N_pos_n_cols + 2] * G_shape_z) / N_pos_shape_z);

		const int group = x + G_shape_y * y + G_shape_y * G_shape_z * z;
        
		// if (n < 10) {
        // printf("\n[%d] (%d, %d, %d), g=%d,", n, x, y, z, group);
        // printf("\n[%d] N_G[%d, :]= [%d, %d, %d]", n, n, N_G[n * 3 + 0], N_G[n * 3 + 1], N_G[n * 3 + 2]); }
        
		const int row_idx = n * N_G_n_cols;

		// assign neuron to location-based group 
		N_G[row_idx + N_G_group_id_col] = group;
		
		// if (n < 10) {
        // printf("\n[%d] N_G[%d, :]= [%d, %d, %d]", n, n, N_G[n * 3 + 0], N_G[n * 3 + 1], N_G[n * 3 + 2]); }
		
		// count group: row <> neuron type (1 or 2), column <> group id
		atomicAdd(&G_neuron_counts[group + G * (N_G[row_idx + N_G_neuron_type_col] - 1)], 1);	
	}
}


void set_G_info(
    const int N, 
    const int G, 
    float* N_pos,
    int* N_G,
	int* G_neuron_counts,
    const int G_shape_x, const int G_shape_y, const int G_shape_z,
    const float N_pos_shape_x, const float N_pos_shape_y, const float N_pos_shape_z,
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
	LaunchParameters launch(N, (void *)set_G_info_);  

	set_G_info_ KERNEL_ARGS2(launch.grid3, launch.block3) (
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
		N_pos_shape_x, N_pos_shape_y, N_pos_shape_z,
		G_neuron_counts
	);
	
	cudaDeviceSynchronize();

    }