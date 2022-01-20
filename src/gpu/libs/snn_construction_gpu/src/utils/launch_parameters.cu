#include <utils/launch_parameters.cuh>


void print_theoretical_occupancy(
	const int block_size, 
	const void* kernel)
{
	int max_active_blocks;
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0);

	int cuda_device;
	cudaDeviceProp cuda_device_props;

	cudaGetDevice(&cuda_device);
	cudaGetDeviceProperties(&cuda_device_props, cuda_device);

	float warpsize = static_cast<float>(cuda_device_props.warpSize); //32
	float max_threads_per_multi_processor = static_cast<float>(cuda_device_props.maxThreadsPerMultiProcessor); //2048
	// float maxThreadsPerBlock = static_cast<float>(cuda_device_props.maxThreadsPerBlock); //1024

	const float occupancy = (
		(static_cast<float>(max_active_blocks) * (static_cast<float>(block_size) / warpsize))
		/ (max_threads_per_multi_processor / warpsize));
	
	printf("\n");

	//printf("block_size: %d \n", block_size);
	//printf("warp_size: %f \n", warp_size);
	//printf("max_threads_per_multi_processor: %f \n", max_threads_per_multi_processor);
	//printf("max_active_blocks: %d\n", max_active_blocks);
	//printf("Theoretical occupancy: %f\n", static_cast<float>(max_active_blocks * (block_size / warp_size)));
	//printf("Theoretical occupancy: %f\n", (max_threads_per_multi_processor / warp_size));
	
	printf("Theoretical occupancy: %f", occupancy);
	printf("\n");
}


LaunchParameters::LaunchParameters()
{
	block_size = 0;
	grid_size = 0;
	block3 = dim3(block_size);
	grid3 = dim3(grid_size);
}

LaunchParameters::LaunchParameters(const int n_threads_x, void*init_func)
{
	func = init_func;
	init_sizes(n_threads_x, init_func);
	block3 = dim3(block_size);
	grid3 = dim3(grid_size);
}

void LaunchParameters::init_sizes(const int n_threads_x, void* init_func)
{
	// initialize grid_size and block_size
	// (optionally) initialize block3 and grid3

	// this functions is reused for 2D initialization
	
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func);
	grid_size = (n_threads_x + block_size - 1) / block_size;
	if (grid_size == 1) {
		block_size = std::min(block_size, n_threads_x);
	}
}

LaunchParameters::LaunchParameters(
	void* init_func, 
	const int n_threads_x, const int n_threads_y,
	const int block_dim_x, const int block_dim_y)
{
	// this functions is reused for reseting the launch parameters
	
	func = init_func;

	init_sizes(n_threads_x * n_threads_y, init_func);

	float block_dim_xf = static_cast<float>(block_dim_x);
	const float block_dim_yf = static_cast<float>(block_dim_y);


	if (n_threads_x > 32)
	{
		while (block_dim_xf > static_cast<float>(n_threads_x) + 32.f)
		{
			block_dim_xf -= 32.f;
		}
	}
	else
	{
		block_dim_xf = static_cast<float>(n_threads_x);
	}

	const float grid_dim_x = ceilf(static_cast<float>(n_threads_x) / block_dim_xf);
	const float grid_dim_y = ceilf(static_cast<float>(n_threads_y) / block_dim_yf);

	block_size = static_cast<int>(block_dim_xf * block_dim_yf);
	grid_size = static_cast<int>(grid_dim_x * grid_dim_y);

	block3 = dim3(static_cast<uint>(block_dim_xf), static_cast<uint>(block_dim_yf));
	grid3 = dim3(static_cast<uint>(grid_dim_x), static_cast<uint>(grid_dim_y));
}


void LaunchParameters::print_info() 
{
	if (block3.y > 1)
	{
		printf("\nblock_size = %d * %d = %d", block3.x, block3.y, block_size);
		printf("\ngrid_size = %d * %d = %d", grid3.x, grid3.y, grid_size);
	}
	else
	{
		printf("\ngrid_size = %d", grid_size);
		printf("\nblock_size = %d", block_size);
	}

	printf("\nmin_grid_size = %d", min_grid_size);
	printf("\n");
	print_theoretical_occupancy(block_size, func);
}

