#include <simulation/snn_simulation.cuh>


__global__ void update_N_state_(
	const int N, 
	const int G,
	const float t,
	curandState* randstate, 
	float* N_pos,
	const int* N_G,
	const float* G_props,
	float* N_states,
	float* fired
	// float* neuron_color,
	// float* debug_i,
	// float* debug_v
)
{
	const int n = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (n < N)
	{
		curandState local_state = randstate[n];
		fired[n] = 0.f;
		N_pos[n * 13 + 10] = .3f;

		const float ntype = __int2float_rn(N_G[n * 2]) - 1;
		const int src_G = N_G[n * 2 + 1];

		float pt = N_states[n];
		float u = N_states[n + N];
		float v = N_states[n + 2 * N];
		const float a = N_states[n + 3 * N];
		const float b = N_states[n + 4 * N];
		const float c = N_states[n + 5 * N];
		const float d = N_states[n + 6 * N];
		float i = N_states[n + 7 * N];

		// printf("\n (%d) (src_G=%d, pt=%f, u=%f, v=%f, a=%f, b=%f, c=%f, d=%f, i=%f)", 
		// n, src_G, pt, u, v, a, b, c, d, i);

		if ((G_props[src_G + G] > 0.f) && (pt > 0.f) && (curand_uniform(&local_state) < pt))
		{
			const float rt = curand_uniform(&local_state);
			// i += (G_props[src_G + 3 * G] * ntype + G_props[src_G + 2 * G] * (1.f - ntype)) * rt;
			i += (G_props[src_G + 3 * G] * ntype + G_props[src_G + 2 * G] * (1.f - ntype)) * rt;
			// printf("G_props[src_G + 3 * G]=%f, G_props[src_G + 2 * G]=%f \n", 
			// 	   G_props[src_G + 3 * G], G_props[src_G + 2 * G]);
		}

		if (v > 30.f)
		{
			v = c;
			u = u + d;
			fired[n] = t;
			N_pos[n * 13 + 10] = 1.f;
		} 
		
		v = v + 0.5f * (0.04f * v * v + 5 * v + 140 - u + i);
		v = v + 0.5f * (0.04f * v * v + 5 * v + 140 - u + i);

		u = u + a * (b * v - u);

		N_states[n + N] = u;
		N_states[n + 2 * N] = v;
		// debug_i[n]  = i;
		// debug_v[n]  = v;
		N_states[n + 7 * N] = 0.f;
		
		randstate[n] = local_state;
	}
}


__global__ void update_voltage_plot_(
	const int* voltage_plot_map,
	const float* N_states,
	float* voltage_plot_data,
	// const int min_idx,
	// const int max_idx,
	const int plot_length,
	const int t,
	const int N,
	const int n_voltage_plots
)
{
	const int plot_idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (plot_idx < n_voltage_plots)
	{
		const int n = voltage_plot_map[plot_idx];
		if (n >= 0){
			const int start_idx = plot_idx * plot_length * 2;
			voltage_plot_data[start_idx + 2 * t + 1] = (
				N_states[n + 2 * N] / 200.f + __int2float_rn(plot_idx) + 0.5 );
		}

	}
}


__global__ void update_scatter_plot_(
	const int* scatter_plot_map,
	const float* fired,
	float* scatter_plot_data,
	// const int min_idx,
	// const int max_idx,
	const int plot_length,
	const int t,
	const int N,
	const int n_scatter_plots
)
{
	const int plot_idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (plot_idx < n_scatter_plots)
	{
		const int n = scatter_plot_map[plot_idx];
		if (n >= 0){
			const int start_idx = plot_idx * plot_length * 13;
			// scatter_plot_data[start_idx + 13 * t + 3] = 0.;
			// scatter_plot_data[start_idx + 13 * t + 4] = 0.;
			// scatter_plot_data[start_idx + 13 * t + 5] = 0.;
			scatter_plot_data[start_idx + 13 * t + 10] = fired[n];

			// scatter_plot_data[start_idx + 13 * t + 5] = (
				// 999.
				// N_states[n + 2 * N] / 2. + __int2float_rn(plot_idx) * 100 + 50 
			// );
		}

	}
}


SnnSimulation::SnnSimulation(
    const int N_,
    const int G_,
    const int S_,
    const int D_,
	const int T_,

	const int n_voltage_plots_,
    const int voltage_plot_length_,
	float* voltage_plot_data_,
	int* voltage_plot_map_,
    
	const int n_scatter_plots_,
    const int scatter_plot_length_,
	float* scatter_plot_data_,
	int* scatter_plot_map_,
	
	curandState* rand_states_,
	float* N_pos_,
	int* N_G_,
    float* G_props_, 
    int* N_rep_, 
    int* N_delays_, 
    float* N_states_,
	float* N_weights_,
	float* fired_,
	float* firing_times_,
	int* firing_idcs_,
	int* firing_counts_
){
    
	N = N_;
	G = G_;
	S = S_;
	D = D_;
	T = T_;

	n_voltage_plots = n_voltage_plots_;
	voltage_plot_length = voltage_plot_length_;
	voltage_plot_data = voltage_plot_data_;
	voltage_plot_map = voltage_plot_map_;

	n_scatter_plots = n_scatter_plots_;
	scatter_plot_length = scatter_plot_length_;
	scatter_plot_data = scatter_plot_data_;
	scatter_plot_map = scatter_plot_map_;

	rand_states = rand_states_;

	N_pos = N_pos_;
	N_G = N_G_;
    G_props = G_props_; 
    N_rep = N_rep_;
    N_delays = N_delays_;
    N_states = N_states_;
	N_weights = N_weights_;

	fired = fired_;
	
	firing_times = firing_times_;
	firing_idcs = firing_idcs_;
	firing_counts = firing_counts_;

	firing_times_write = firing_times;
	firing_times_read = firing_times;

	firing_idcs_write = firing_idcs;
	firing_idcs_read = firing_idcs;
	
	firing_counts_write = firing_counts;

	reset_firing_times_ptr_threshold = 13 * N;

    lp_update_state = LaunchParameters(N, (void *)update_N_state_);
    lp_update_voltage_plot = LaunchParameters(n_voltage_plots, (void *)update_voltage_plot_);
	lp_update_scatter_plot = LaunchParameters(n_scatter_plots, (void *)update_scatter_plot_);

    // fired
	checkCusparseErrors(cusparseCreate(&fired_handle));
	checkCusparseErrors(cusparseCreateDnMat(&firing_times_dense,
		1, N, N,
		fired,
		CUDA_R_32F, CUSPARSE_ORDER_ROW));
	
	checkCusparseErrors(cusparseCreateCsr(&firing_times_sparse, 1, N, 0,
		firing_counts_write,
		firing_idcs_write,
		firing_times_write,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	// allocate an external buffer if needed
	checkCusparseErrors(cusparseDenseToSparse_bufferSize(
		fired_handle, firing_times_dense, firing_times_sparse,
		CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
		&fired_buffer_size));
	checkCudaErrors(cudaMalloc(&fired_buffer, fired_buffer_size));
	
}


__global__ void update_current_(
	const int N,
	const int S,
	const int D,
	const int* fired_idcs_read, 
	const int* fired_idcs, 
	const float* firing_times_read,
	const float* firing_times,
	const int* N_rep, 
	const float* N_weights, 
	float* N_states,
	const int n_fired_m1_to_end,
	const int n_fired,
	const int t, 
	const int* N_delays
	//float* debug_i
)
{
	//const int tid_x = blockIdx.get_x * blockDim.get_x + threadIdx.get_x;
	//const int fired_idx = start_idx + blockIdx.get_x * blockDim.get_x + threadIdx.get_x;
	const int fired_idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (fired_idx < n_fired)
	{
		int n;
		int firing_time;

		if (fired_idx < n_fired_m1_to_end)
		{
			// global index of firing-array < len(fired-array) 
			// -> use the trailing pointer
			n = fired_idcs_read[fired_idx];
			firing_time = __int2float_rn(firing_times_read[fired_idx]);
		}
		else
		{
			// global index of firing-array >= len(fired-array) 
			// -> use the 'normal' pointer
			n = fired_idcs[fired_idx - n_fired_m1_to_end];
			firing_time = __int2float_rn(firing_times[fired_idx - n_fired_m1_to_end]);
		}
		// const int firing_time = __int2float_rn(firing_times[fired_idx]);

		// const int delay_idx = n * (D + 1) + (t - firing_time);

		// N_delays.shape = (D+1, N)
		// t - firing-time = delay -> use the dealy to infer which synapses must be activated using
		// the N_delays array. 
		const int delay_idx = n + N * (t - firing_time);
		const int syn_idx_start = n * S + N_delays[delay_idx];
		const int syn_idx_end = n * S + N_delays[delay_idx + N];

	 	// if ((syn_idx_end > N * S) || ((t - firing_time) >= D))
		// {
	 	// 	printf("\n (%d)-(%d < %d = %d) (n_fired=%d) t:%d, firing_time: %d (%d), delay_idx: %d, delay_count: %d, syn = {%d, ..., d}", 
		// 	 n, fired_idx, n_fired_m1_to_end, 
		// 	 fired_idx < n_fired_m1_to_end, 
		// 	 n_fired,
	 	// 	 t,
		// 	 firing_time, t-firing_time, delay_idx, N_delays[delay_idx + N], 
		// 	 N_rep[syn_idx_start] 
		// 	// N_rep[syn_idx_end -1]
		// 	);
	 	// 	printf("\n (%d) syn_idx_start: %d, syn_idx_end: %d",n, syn_idx_start, syn_idx_end);
		// 	// printf("\nfiring_times:");
		// 	// for (int i = 0; i < 15; i++) {
		// 	// 	printf("\n");
		// 	// 	for (int j = 0; j < N; j++) {
		// 	// 		printf("%.0f, ", firing_times[i * N + j]);
		// 	// 	}
		// 	// }
	 	// }


	
		for (int s = syn_idx_start; s < syn_idx_end; s++)
		{
			// if (N_rep[s] >= 0){

			// if (s >= N * S) 
			// 	printf("\n (%d)->(%d) [I] = %f", n, N_rep[s], N_states[N_rep[s] + 7 * N]);
			// atomicAdd(&debug_i[N_rep[s]], N_weights[s]);
			
			// if (s < N * S) 
			atomicAdd(&N_states[N_rep[s] + 7 * N], N_weights[s]);
			// if (s >= N * S)
			// 	printf("\n (%d)->(%d) [I] = %f", n, N_rep[s], N_states[N_rep[s] + 7 * N]);

			// } else {
			// 	printf("\n (%d) N_rep[s] = %d", n, N_rep[s]);
			// }
		}
	}
	
}


void SnnSimulation::update_voltage_plot()
{

	update_voltage_plot_ KERNEL_ARGS2(lp_update_voltage_plot.grid3, 
									  lp_update_voltage_plot.block3) (
		voltage_plot_map,
		N_states,
		voltage_plot_data,
		// voltage_plot_info.h_M[0],
		// voltage_plot_info.h_M[2],
		voltage_plot_length,
		t % voltage_plot_length,
		N,
		n_voltage_plots
	);
	
	update_scatter_plot_ KERNEL_ARGS2(lp_update_scatter_plot.grid3, 
									  lp_update_scatter_plot.block3) (
		scatter_plot_map,
		fired,
		scatter_plot_data,
		// voltage_plot_info.h_M[0],
		// voltage_plot_info.h_M[2],
		scatter_plot_length,
		t % scatter_plot_length,
		N,
		n_scatter_plots
	);

	// k_update_scatter_plot_data KERNEL_ARGS2(l_update_scatter_plot_data.grid3,
	// 								        l_update_scatter_plot_data.block3) (
	// 	firing_plot_map.get_dp(),
	// 	fired.get_dp(),
	// 	renderer->firing_plot.ebo_data.d,
	// 	firing_plot_info.h_M[0],
	// 	firing_plot_info.h_M[2],
	// 	renderer->firing_plot.plot_length,
	// 	t % renderer->firing_plot.plot_length,
	// 	N,
	// 	renderer->firing_plot.restart_index_
	// );
}

// void print_array()

void SnnSimulation::print_info(bool bprint_idcs){
	std::cout << "\n\n  ------------------------------------ ";
	printf("\nt=%d", t);
	printf("\nn_fired=%d", n_fired);
	printf("\nn_fired_m1_to_end=%d", n_fired_m1_to_end);
	printf("\nn_fired_0=%d", n_fired_0);
	printf("\nn_fired_m1=%d", n_fired_m1);
	printf("\nn_fired_total=%d", n_fired_total);
	printf("\nn_fired_total_m1=%d", n_fired_total_m1);
	printf("\nfiring_counts_write=%p", (void * )firing_counts_write);
	printf("\n");
	
	if (bprint_idcs){
		printf("\nfiring_idcs:");
		for (int i = 0; i < 15; i++) {
			printf("\n");
			for (int j = 0; j < N; j++) {
				int firing_index;
				cudaMemcpy(&firing_index, firing_idcs + i * N + j, 
					sizeof(float), cudaMemcpyDeviceToHost);
				printf("%d, ", firing_index);
			}
		}
		printf("\n");
	}

	printf("\nfiring_times:");
	for (int i = 0; i < 15; i++) {
		printf("\n");
		for (int j = 0; j < N; j++) {
			float firing_time;
			cudaMemcpy(&firing_time, firing_times + i * N + j, 
				sizeof(float), cudaMemcpyDeviceToHost);
			printf("%.0f, ", firing_time);
		}
	}
	printf("\n");


}
void SnnSimulation::update(const bool verbose)
{	
	// if (verbose)
	// {
	// 	std::cout << "\n\n  ------------------------------------ ";
	// 	std::cout << "( " << t << " ) ------------------------------------\n";
	// }

	// renderer->neurons_bodies.pos_colors.map_buffer();

	update_N_state_ KERNEL_ARGS2(lp_update_state.grid3, lp_update_state.block3 )(
		N,
		G,
		static_cast<float>(t),
		rand_states,
		N_pos,
		N_G,
		G_props,
		N_states,
		fired
		// debug_i.get_dp(),
		// debug_v.get_dp()
    );

	// renderer->neurons_bodies.pos_colors.unmap_buffer();

	// if (verbose)
	// {
	// 	std::cout << "\n" << t << "\n";
	// 	neuron_states.print_d_m();
	// 	fired.print_d_m();
	// 	neuron_groups.print_d_m();
	// }

	if (b_update_voltage_plot)
	{
		update_voltage_plot();
	}

	checkCudaErrors(cudaDeviceSynchronize());

	// fired
	checkCusparseErrors(cusparseDenseToSparse_analysis(
		fired_handle, firing_times_dense, firing_times_sparse,
		CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, fired_buffer));
	
	checkCudaErrors(cudaDeviceSynchronize());

	checkCusparseErrors(cusparseDenseToSparse_convert(
		fired_handle, firing_times_dense, firing_times_sparse,
		CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, fired_buffer));

	// if (verbose)
	// {
		
	// 	// debug_i.print_d_m();
	// 	// debug_v.print_d_m();
	// 	printf("\n");
	// 	printf("\nn_fired_m1=%d", n_fired_m1);
	// 	printf("\nn_fired_total=%d", n_fired_total);
	// 	printf("\nn_fired_0=%d", n_fired_0);
	// 	printf("\nn_fired_total_m1=%d", n_fired_total_m1);
	// 	printf("\nfiring_counts_write=%p", (void * )firing_counts_write);
	// 	printf("\n");
	// 	// firing_idcs_write.print_d_m();
	// }

	checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(
		&n_fired_0, firing_counts + firing_counts_idx, sizeof(int), cudaMemcpyDeviceToHost));
	// n_fired_0 = firing_counts.d_M[firing_counts_idx];

	// bool print_cond = (t==15);
	// bool print_cond = resetting;
	// bool print_cond = false;
	
	// if (print_cond){
	// 	print_info();
	// }

	n_fired_total += n_fired_0;
	n_fired += n_fired_0;
	firing_counts_idx += 2;

	if (n_fired_total > n_fired_total_m1) {
		n_fired_m1_to_end += n_fired_0;
	}

	if (t >= D)
	{
		cudaMemcpy(&n_fired_m1, firing_counts + firing_counts_idx_m1, 
                   sizeof(int), cudaMemcpyDeviceToHost);

        //n_fired_m1 = firing_counts.d_M[firing_counts_idx_m1];
		n_fired_total_m1 += n_fired_m1;
		n_fired -= n_fired_m1;
		n_fired_m1_to_end -= n_fired_m1;
		firing_counts_idx_m1 += 2;
	}

	if (n_fired_total <= reset_firing_times_ptr_threshold)
	{
		firing_times_write += n_fired_0;
		firing_idcs_write += n_fired_0;
		firing_counts_write += 2;
	}
	else
	{
		firing_times_write = firing_times;
		firing_idcs_write = firing_idcs;
		firing_counts_write = firing_counts;
		firing_counts_idx = 1;
		n_fired_total = 0;
		// printf("\nt: %d (reset)\n", t);
		resetting = true;
	}

	if (n_fired_total_m1 <= reset_firing_times_ptr_threshold)
	{
		firing_times_read += n_fired_m1;
		firing_idcs_read += n_fired_m1;
	}
	else
	{
		firing_times_read = firing_times;
		firing_idcs_read = firing_idcs;

		n_fired_m1_to_end = n_fired_total;
		firing_counts_idx_m1 = 1;
		n_fired_total_m1 = 0;
		// printf("\nt: %d (m1-reset)\n", t);
		resetting = false;
		// print_info();
	}

	// if (print_cond){
	// 	print_info();
	// }

	
	int block_dim_x = 32;
	int grid_dim_x = static_cast<int>(::ceilf(static_cast<float>(n_fired) / static_cast<float>(block_dim_x)));

	checkCudaErrors(cudaDeviceSynchronize());
	
	update_current_ KERNEL_ARGS2(grid_dim_x, block_dim_x)(
		N,
		S,
		D,
		firing_idcs_read,
		firing_idcs,
		firing_times_read,
		firing_times,
		N_rep,
		N_weights,
		N_states,
		n_fired_m1_to_end,
		n_fired,
		t,
		N_delays
		// debug_i.get_dp()
    );
	
	checkCudaErrors(cudaDeviceSynchronize());

	t++;

	cusparseCsrSetPointers(firing_times_sparse,
                           firing_counts_write,
	                       firing_idcs_write,
	                       firing_times_write);

	// if (true)
	// {
	// 	debug_i.print_d_m();
	// 	//neuron_states.print_d_m();
	// }

	checkCudaErrors(cudaDeviceSynchronize());
	
	// if (verbose)
	// {
	// 	printf("\n");
	// }
}



