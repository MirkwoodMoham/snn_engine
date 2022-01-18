#include <snn_construction.cuh>


void __global__ k_set_neuron_location_group(
	const int N,
    const int G,
	const float* N_pos, 
    const int N_pos_stride,
	int* N_G,

	//int* location_groups_neuron_counts,  // NOLINT(readability-non-const-parameter)
	const float G_shape_x,
	const float G_shape_y,
	const float G_shape_z,

	const float max_pos_x,
	const float max_pos_y,
	const float max_pos_z
)
{
	const int n = blockIdx.x * blockDim.x + threadIdx.x;  // NOLINT(bugprone-narrowing-conversions)
	
	if (n < N)
	{
		const int x = __float2int_rd((N_pos[n * N_pos_stride] * G_shape_x) / max_pos_x);  
		const int y = __float2int_rd((N_pos[n * N_pos_stride + 1] * G_shape_y) / max_pos_y);
		const int z = __float2int_rd((N_pos[n * N_pos_stride + 2] * G_shape_z) / max_pos_z);

		const int group = x
			+ G_shape_y * y
			+ G_shape_y * G_shape_z * z;
        
            
        printf("\n[%d] (%d, %d, %d), g=%d,", n, x, y, z, group);

        printf("\n[%d] N_G[%d, :]= [%d, %d, %d]", n, n, N_G[n * 3 + 0], N_G[n * 3 + 1], N_G[n * 3 + 2]);
        N_G[n * 3 + 2] = group;
        printf("\n[%d] N_G[%d, :]= [%d, %d, %d]", n, n, N_G[n * 3 + 0], N_G[n * 3 + 1], N_G[n * 3 + 2]);

		// const int group_idx_offset = G * (N_G[n]-1);
		// atomicAdd(&location_groups_neuron_counts[group_idx_offset + group], 1);	
	}
}


void init_pos_gpu(
    int N, 
    int G, 
    float* N_pos,
    int* N_G,
    int G_shape_x, int G_shape_y, int G_shape_z
)
{	

	cudaDeviceSynchronize();
	LaunchParameters launch(N, (void *)k_set_neuron_location_group);  

	k_set_neuron_location_group KERNEL_ARGS2(launch.grid3, launch.block3) (
		N,
        G,
		N_pos,
        13,
		N_G,
		// LG_neuron_counts.dp,
		static_cast<float>(G_shape_x),
		static_cast<float>(G_shape_y),
		static_cast<float>(G_shape_z),
		1.f,
		1.f,
		1.f);
	
	cudaDeviceSynchronize();

    }