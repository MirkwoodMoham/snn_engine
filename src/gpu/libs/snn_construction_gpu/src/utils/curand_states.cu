#include <utils/curand_states.cuh>


__forceinline__ __device__ int random_uniform_int(curandState *local_state, const float min, const float max)
{
	return __float2int_rd(fminf(min + curand_uniform(local_state) * (max-min +1.f), max));
}


__global__ void print_random_numbers_(curandState* state,
	const int n_curand_states)
{
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < n_curand_states) && (row < 10)) {
		curandState local_state = state[row];
		printf("%f\n", curand_uniform(&local_state));
		//printf("%d\n", random_uniform_int(&local_state, 0., 10.));
	}
}


void print_random_numbers(curandState* states, const int n_states){

	LaunchParameters launch(std::max(n_states, 9), (void*)print_random_numbers_);
	print_random_numbers_ KERNEL_ARGS2(launch.grid3, launch.block3) (states, n_states);
}


void print_random_numbers2(std::shared_ptr<CuRandStates> p){

	print_random_numbers(p->states, p->n_states);
}


__global__ void initialise_curand_k(curandState* state, const unsigned long long int* seed,
	const int n_curand_states)
{
	const int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n_curand_states) {
		curand_init(seed[row], 0, 0, &state[row]);
	}
}


CuRandStates::CuRandStates(
	const int n,
    bool verbose
){
	if (verbose) printf("\n(curand) (0/5) starting initialization of %d curand-states .. \n", n);
	
    n_states = n;
	// p = (long long int*) = &states;
	
	if (verbose) printf("(curand) (1/5) allocating GPU-memory (1) .. \n");
	unsigned long long int* d_seeds = 0;
	checkCudaErrors(cudaMalloc(&states, n_states * sizeof(curandState)));
	if (verbose) printf("(curand) (2/5) allocating GPU-memory (2) .. \n");
	checkCudaErrors(cudaMalloc((void**)&d_seeds, n_states * sizeof(unsigned long long int)));

	if (verbose) printf("(curand) (3/5) generating sequence ..\n");

	LaunchParameters launch(n_states, (void*)initialise_curand_k);
	std::seed_seq seq({ rand(),rand(),rand() });
    std::vector<std::uint32_t> seeds(n);
    seq.generate(seeds.begin(), seeds.end());
	
	if (verbose) printf("(curand) (4/5) copying ..\n");

	if (n_states <= 300 * 1000)
	{
		cudaMemcpy(d_seeds, &seeds[0], n_states * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
	} 
	else 
	{
		std::uint32_t* ptr = &seeds[0];
		int n_copied = 0;
		int batch_size = 25 * 1000;

		std::string msg;
		if (verbose) {
			msg = "copied: 0/" + std::to_string(n_states);
			std::cout << msg;
		}

		while (n_copied < n_states){
			
			if (n_copied + batch_size > n_states){
				batch_size = n_states - n_copied;
			} 
			cudaMemcpy(d_seeds, ptr, batch_size * sizeof(unsigned long long int), cudaMemcpyHostToDevice);
			n_copied += batch_size;
			ptr += batch_size;
			if (verbose) { 
				std::cout << std::string(msg.length(),'\b');
				msg = "copied: " + std::to_string(n_copied) + "/" + std::to_string(n_states);
				std::cout << msg;
			}
		}
		printf("\n");
	}
	
	if (verbose) printf("(curand) (5/5) initializing ..\n");
	initialise_curand_k KERNEL_ARGS2(launch.grid3, launch.block3) (states, d_seeds, n_states);
	
	checkCudaErrors(cudaDeviceSynchronize());
	
	cudaFree(d_seeds);
	
	// print_random_numbers(states, n_states);

}

