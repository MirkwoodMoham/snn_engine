#pragma once

#include <utils/cpp_header.h>

void print_theoretical_occupancy(
	int block_size, 
	const void* kernel
);


struct LaunchParameters
{
	dim3 block3;
	dim3 grid3;

	int block_size;			// The launch configurator returned block size 
	int min_grid_size;		// The minimum grid3 size needed to achieve the 
							// maximum occupancy for a full device launch 
	int grid_size;			// The actual grid3 size needed, based on input size

	void* func;

	LaunchParameters();
	
	LaunchParameters(
		int n_threads_x, 
		void* init_func,
		int dynamicSMemSize = 0,
		int blockSizeLimit = 0
	);
	void init_sizes(
		int n_threads_x, 
		void* init_func,
		int dynamicSMemSize = 0,
		int blockSizeLimit = 0
	);

	// LaunchParameters(
	// 	int n_threads_x, int n_threads_y, 
	// 	void* init_func, 
	// 	int block_dim_x = 128, int block_dim_y = 1);

	void print_info();
};