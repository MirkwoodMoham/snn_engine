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
			i += (G_props[src_G + 3 * G] * ntype + G_props[src_G + 2 * G] * (1.f - ntype)) * rt;
			// printf("G_props[src_G + 3 * G]=%f, G_props[src_G + 2 * G]=%f \n", 
			// 	   G_props[src_G + 3 * G], G_props[src_G + 2 * G]);
		}
		
		if (G_props[src_G + 7 * G] > 0.f)
		{
			float input_type = G_props[src_G];	
			if (input_type >= 0.){
				i += G_props[src_G + 9 * G] * input_type + G_props[src_G + 8 * G] * (1.f - input_type);
			}
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
	const int G,
	const int S,
	const int D,
	const int* fired_idcs_read, 
	const int* fired_idcs, 
	const float* firing_times_read,
	const float* firing_times,
	const int* N_G,
	const float* G_props,
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
		// const int syn_idx_start = n * S + N_delays[delay_idx];
		// const int syn_idx_end = n * S + N_delays[delay_idx + N];

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

		int snk_neuron;
		int snk_G;
		// const int src_G = N_G[n * 2 + 1];
		bool snk_G_is_sensory = false;
		//bool src_G_is_sensory = G_props[src_G + 7 * G] > 0.f;
	
		// for (int s = syn_idx_start; s < syn_idx_end; s++)
		// {
			// if (N_rep[s] >= 0){

			// if (s >= N * S) 
			// 	printf("\n (%d)->(%d) [I] = %f", n, N_rep[s], N_states[N_rep[s] + 7 * N]);
			// atomicAdd(&debug_i[N_rep[s]], N_weights[s]);
			
			// if (s < N * S) 

			// snk_neuron = N_rep[s];
			// snk_G = N_G[snk_neuron * 2 + 1];
			// snk_G_is_sensory = G_props[snk_G + 7 * G] > 0.f;
			// if (!snk_G_is_sensory)
			// {
			// 	atomicAdd(&N_states[snk_neuron + 7 * N], N_weights[s]);
			// }
			
			// atomicAdd(&N_states[N_rep[s] + 7 * N], N_weights[s]);
			// if (s >= N * S)
			// 	printf("\n (%d)->(%d) [I] = %f", n, N_rep[s], N_states[N_rep[s] + 7 * N]);

			// } else {
			// 	printf("\n (%d) N_rep[s] = %d", n, N_rep[s]);
			// }
		//}

		int idx;
		int row_end = N_delays[delay_idx + N]; 
		for (int row = N_delays[delay_idx]; row < row_end; row++)
		{
			idx = n + N * row;
			snk_neuron = N_rep[idx];
			snk_G = N_G[snk_neuron * 2 + 1];
			snk_G_is_sensory = G_props[snk_G + 7 * G] > 0.f;
			if (!snk_G_is_sensory)
			{
				atomicAdd(&N_states[snk_neuron + 7 * N], N_weights[idx]);
			}
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
		G,
		S,
		D,
		firing_idcs_read,
		firing_idcs,
		firing_times_read,
		firing_times,
		N_G,
		G_props,
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


__device__ void roll_copy(
	
	int* write_array, int* read_array, 
	int write_col, int read_col, 
	int write_row_start, 
	int n_write_array_cols, int n_read_array_cols,
	const int copy_length, 
	const int read_offset, bool bprint){
	int write_idx;
	int read_idx;
	//int roll_mod = abs(swap_snk_N_s_start - swap_src_N_s_start);

	for (int s=0; s < copy_length; s++){

		write_idx = write_col + (write_row_start + s) * n_write_array_cols;
		read_idx = read_col + (write_row_start + ((s +  read_offset) % copy_length)) * n_read_array_cols;

		if (bprint){
			printf("\nwrite_array[%d, %d]=%d -> read_array[%d, %d]=%d", 
			write_row_start + s, write_col, 
			write_array[write_idx], 
			write_row_start + ((s +  read_offset) % copy_length), read_col,
			read_array[read_idx] );
		}

		write_array[write_idx] = read_array[read_idx];

	}	

}


__device__ void shift_values_row_wise_(
	int shift_,
	int* array0, int* array1,
	int col0, int col1,
	int n_cols0, int n_cols1,
	int end_row
){
	int idx_end0 = col0 + (end_row) * n_cols0;
	int idx_end1 = col1 + (end_row) * n_cols1;
	int value0;
	for (int k=-shift_; k < 0; k++){
		value0 = array0[idx_end0 + (k+1) * n_cols0];
		if (value0 > 0){
			array0[idx_end0 + k * n_cols0] = value0;
			array1[idx_end1 + k * n_cols1] = array1[idx_end1 + (k+1) * n_cols1];
		}
		
	}
}


__device__ void generate_synapses(
	int N,
	int n,
	int neuron_idx,
	int* N_rep,
	int* G_swap_tensor,
	int& swap_src_N_s_start, int& swap_snk_N_s_start,
	int& swap_src_G_count, int& swap_snk_G_count,
	int max_snk_count,
	curandState &local_state,
	int G_swap_tensor_shape_1, 
	int swap_type,
	int index_offset,
	int relative_index_offset,
	bool bprint
){
	
	int snk_N;
	int min_G_swap_snk = G_swap_tensor[neuron_idx + swap_snk_N_s_start * G_swap_tensor_shape_1];
	int max_G_swap_snk = G_swap_tensor[neuron_idx + (swap_snk_N_s_start + swap_snk_G_count - 1) * G_swap_tensor_shape_1];
	if (swap_snk_G_count == 0){
		min_G_swap_snk = max_snk_count;
		max_G_swap_snk = -1;
	}
	//int s_offset = 0;
	float r;
	// int max_snk_count = (snk_type - 1) * total_snk_G_count_exc + (2 - snk_type) * total_snk_G_count_inh;

	int s_end = swap_src_N_s_start + swap_src_G_count;

	for (int s=swap_src_N_s_start; s < s_end; s++){
		
		r = curand_uniform(&local_state);

		// snk_N = -N_rep[n + (s-s_offset) * N] - total_src_G_count;
		// snk_type = N_G[snk_N * 2];

		// snk_N = __float2int_rd(r * __int2float_rn(max_snk_count)) + (snk_type - 1) * total_snk_G_count_inh;
		snk_N = __float2int_rd(r * __int2float_rn(max_snk_count)) + relative_index_offset;
			
				
		if (bprint) printf("\n[%d, %d] new=%d (%f), t=%d, s=%d, [%d, %d], [offset = %d - %d]", 
						   n, neuron_idx, snk_N, r, swap_type, s,
						   min_G_swap_snk, max_G_swap_snk, 
						   index_offset, relative_index_offset);

		if (swap_snk_G_count < max_snk_count)
		{	
			bool found = false;
			int i = 0;	
			int j = 0;
			int swap_idx = neuron_idx + (swap_snk_N_s_start)  * G_swap_tensor_shape_1;
			int G_swap0;
			int G_swap_m1;

			int write_row;
			// int write;

			int last_write_mode = 0;
			int write_mode = 0;

			while ((!found) && (j < 40)){
				
				write_mode = 0;
				//write_row = s - s_offset;
				// write = -i;
				swap_idx = neuron_idx + (swap_snk_N_s_start + i )  * G_swap_tensor_shape_1;
				
				G_swap0 = G_swap_tensor[swap_idx];
				G_swap_m1 = G_swap_tensor[swap_idx - G_swap_tensor_shape_1];


				if((snk_N < min_G_swap_snk) || (swap_snk_G_count == 0)){
				
					if (swap_snk_G_count == 0){
						max_G_swap_snk = snk_N;
					}
					min_G_swap_snk = snk_N;
					write_row = swap_snk_N_s_start - 1;
					write_mode = 1;
					// G_swap_tensor[swap_idx - G_swap_tensor_shape_1] = G_swap0;	
					// G_swap_tensor[swap_idx] = snk_N;		
				}
				else if((snk_N > max_G_swap_snk)){
					write_mode = 2;
					if (swap_snk_G_count == 0){
						min_G_swap_snk = snk_N;
					}
					max_G_swap_snk = snk_N;
					write_row = swap_snk_N_s_start + swap_snk_G_count - 1;

				}
				else if ((G_swap_m1 < snk_N) && (snk_N < G_swap0)){
					write_mode = 3;
					write_row = swap_snk_N_s_start + i - 1;
				}

				found = write_mode > 0;

				if (found){
					swap_snk_N_s_start -= 1;
					// write = snk_N;
					// s_offset++;
					swap_snk_G_count++;
					// G_swap_tensor[neuron_idx + (write_row) * G_swap_tensor_shape_1] = write;
					break;}
				

				if ((snk_N == G_swap0)){
					snk_N = (snk_N + 1) % max_snk_count;
				}

				if (bprint || (j >= 30)) {
					printf("\n[%d, %d] + new=%d[i=%d, write_mode=%d] G_swap_m1=%d, G_swap0=%d, [%d, %d], max_snk_count=%d, (%d), swap_snk_G_count=%d, s=%d", 
						n, neuron_idx, snk_N, i, write_mode, G_swap_m1, G_swap0, 
						min_G_swap_snk,
						max_G_swap_snk,
						max_snk_count, swap_type, swap_snk_G_count, s);
				}
				
				i = (i + 1) % swap_snk_G_count;
				j++;

				// if (j >= 10){
				// 	printf("\nn=%d; new=%d[%d] G_swap0=%d, max_snk_count=%d, (%d), s=%d", 
				// 		   n, snk_N, i, G_swap0, max_snk_count, swap_type, s);
				// }
			}

			if (bprint || (j >= 30)) {
				printf("\n[%d, %d] (found j=%d, mod:%d->%d) N_rep[%d, %d]=%d (%d) [%d (snk_N) + %d - %d]", 
					n, neuron_idx, j, last_write_mode, write_mode,
					write_row, n, N_rep[n + (write_row) * N], N_rep[n + (swap_snk_N_s_start-1) * N], snk_N,
					index_offset, relative_index_offset);
			}

			//|| (j >= 30)
			if (write_mode > 1){
				shift_values_row_wise_(
					write_row - swap_snk_N_s_start + 1,
					N_rep, G_swap_tensor,
					n, neuron_idx,
					N, G_swap_tensor_shape_1,
					write_row);
			}

			N_rep[n + (write_row) * N] = snk_N + index_offset - relative_index_offset;
			G_swap_tensor[neuron_idx + (write_row) * G_swap_tensor_shape_1] = snk_N;

			if (bprint || (j >= 30)) {
				printf("\n[%d, %d] (found j=%d, mod:%d->%d) N_rep[%d, %d]=%d (%d) [%d (snk_N) + %d - %d]", 
					n, neuron_idx, j, last_write_mode, write_mode,
					write_row, n, N_rep[n + (write_row) * N], N_rep[n + (swap_snk_N_s_start-1) * N], snk_N,
					index_offset, relative_index_offset);
			}
			last_write_mode = write_mode;
		} 

		// 	swap_snk_G_count++;
	}

}





__global__ void swap_groups_(
	const long* neurons, const int n_neurons, 
	const long* groups, const int n_groups,
	const int* group_indices,
	int* G_swap_tensor, const int max_neurons_per_group, const int G_swap_tensor_shape_1,
	const float* swap_rates,
	const int* group_neuron_counts_inh, const int* group_neuron_counts_exc, const int* group_neuron_counts_total, 
	const int swap_delay,
	const int* N_relative_G_indices, const int* G_neuron_typed_ccount,
	int N,
	int G,
	int S,
	int* N_G,
	int* N_rep,
	int* N_delays,
	curandState* randstates,
	int* neuron_group_counts,
	const int print_idx
){
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (neuron_idx < n_neurons){

		bool bprint = (neuron_idx == min(print_idx, n_neurons- 1));

		const int n = neurons[neuron_idx];
		
		// the group index contains the 'source group'-column (src_G) of the groups[]-array.
		const int snk_group_index = group_indices[neuron_idx] + 2 * n_groups;

		const int swap_src_G = groups[snk_group_index - 2 * n_groups];
		const int src_G = groups[snk_group_index - n_groups];
		const int swap_snk_G = groups[snk_group_index];

		const int total_src_G_count = group_neuron_counts_total[snk_group_index - 2 * n_groups];
		const int total_snk_G_count = group_neuron_counts_total[snk_group_index];

		if (bprint){		
			printf("\nswap_src %d (%d), src_G %d %d (%d), swap_snk %d (%d)  group_indices[%d] = %d\n", 
			swap_src_G, total_src_G_count,
			N_G[2 * n + 1], src_G, group_neuron_counts_total[snk_group_index - n_groups],
			swap_snk_G, total_snk_G_count, neuron_idx, (int)group_indices[neuron_idx]);
		}

		int snk_N;
		int snk_type;
		int snk_G;


		int s_start = N_delays[n + (swap_delay) * N];
		int s_end =  N_delays[n + (swap_delay + 1) * N];

		int swap_src_N_s_start_inh = s_start;
		int swap_src_N_s_start_exc = s_start;
		int swap_snk_N_s_start_inh = s_start;
		int swap_snk_N_s_start_exc = s_start;

		int swap_src_G_count_inh = 0;
		int swap_src_G_count_exc = 0;
		int swap_snk_G_count_inh = 0;
		int swap_snk_G_count_exc = 0;


		for (int s=s_start; s < s_end; s++)
		{
			snk_N = N_rep[n + s * N];
			snk_type = N_G[snk_N * 2];
			snk_G = N_G[snk_N * 2 + 1];

			if (snk_G == swap_src_G)
			{
				if (snk_type == 1){
					if (swap_src_G_count_inh == 0){
						swap_src_N_s_start_inh = s;
					}
					if (swap_snk_G_count_inh ==0){
						swap_snk_N_s_start_inh = s;
					}
					swap_src_G_count_inh += 1;

				} 
				else if ((snk_type == 2)){
					if (swap_src_G_count_exc == 0){
						swap_src_N_s_start_exc = s;
					}
					if (swap_snk_G_count_exc ==0){
						swap_snk_N_s_start_exc = s;
					}
					swap_src_G_count_exc += 1;
				}
				
				G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = -total_src_G_count-N_rep[n + s * N]; //-2;
				if (bprint) printf("\n(%d) n_snk=%d, (snk_G=%d)  (s=%d) %d %d, src_counts=[%d, %d]", 
					n, N_rep[n + s * N], snk_G, s, snk_G == swap_src_G, snk_G == swap_snk_G, 
					swap_src_G_count_inh, swap_src_G_count_exc);
				
				// N_rep[n + s * N] = N_rep[n + (s + 1) * N];
				N_rep[n + s * N] = -1;
				//swap_src_N_s_start++;
				
			}
			else if (snk_G == swap_snk_G)
			{
				if (snk_type == 1){
					if (swap_snk_G_count_inh == 0){
						swap_snk_N_s_start_inh = s;
					}
					swap_snk_G_count_inh += 1;
				} 
				else if ((snk_type == 2)){
					if (swap_snk_G_count_exc == 0){
						swap_snk_N_s_start_exc = s;
					}
					swap_snk_G_count_exc += 1;
				}

				//G_swap_tensor[neuron_idx + (max_neurons_per_group + N_relative_G_indices[snk_N]) * G_swap_tensor_shape_1] = snk_N;
				G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = N_relative_G_indices[snk_N];
				
				if (bprint) printf("\n(%d) n_snk=%d, (snk_G=%d)  (s=%d) %d %d, snk_N_rel=%d", 
					n, N_rep[n + s * N], snk_G, s, 
					snk_G == swap_src_G, snk_G == swap_snk_G, 
					N_relative_G_indices[snk_N]);
			} 
			else if((swap_src_G_count_inh > 0) 
					|| (swap_src_G_count_exc > 0) 
					|| (swap_snk_G_count_inh > 0)
					|| (swap_snk_G_count_exc > 0)
				){
				G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = N_rep[n + s * N];	
			}

		}

		if (bprint){
			printf("\n\ninh: src=[%d, +%d] snk=[%d, +%d]\n", 
				   swap_src_N_s_start_inh, swap_src_G_count_inh, swap_snk_N_s_start_inh, swap_snk_G_count_inh);
				   printf("exc: src=[%d, +%d] snk=[%d, +%d]\n", 
				   swap_src_N_s_start_exc, swap_src_G_count_exc, swap_snk_N_s_start_exc, swap_snk_G_count_exc);
		}

		if (swap_src_G_count_inh > 0){

			int distance_inh = min((swap_src_G_count_inh + swap_src_N_s_start_inh) - swap_snk_N_s_start_inh, 
								    max(0, swap_src_N_s_start_inh - (swap_snk_G_count_inh + swap_snk_N_s_start_inh)));
			
			if (distance_inh != 0){

				int direction = -1 * (distance_inh > 0) + (distance_inh < 0);

				roll_copy(
					N_rep, G_swap_tensor, 
					n, neuron_idx, 
					swap_src_N_s_start_inh, 
					N, G_swap_tensor_shape_1, 
					abs(swap_snk_N_s_start_inh - swap_src_N_s_start_inh), 
					direction * swap_src_G_count_inh, 
					false);

				swap_src_N_s_start_inh += -distance_inh;

				if (bprint) {printf("\n\nswap_src_N_s_start_inh=%d, distance_inh=%d\n", swap_src_N_s_start_inh, distance_inh);}

			}
		}

		if (swap_src_G_count_exc > 0){

			int distance_exc = min((swap_src_G_count_exc + swap_src_N_s_start_exc) - swap_snk_N_s_start_exc, 
									max(0, swap_src_N_s_start_exc - (swap_snk_G_count_exc + swap_snk_N_s_start_exc)));

			if (distance_exc != 0){
				int direction = -1 * (distance_exc > 0) + (distance_exc < 0);

				roll_copy(
					N_rep, G_swap_tensor, 
					n, neuron_idx, 
					swap_src_N_s_start_exc, 
					N, G_swap_tensor_shape_1, 
					abs(swap_snk_N_s_start_exc - swap_src_N_s_start_exc), 
					direction * swap_src_G_count_exc, 
					false);

					swap_src_N_s_start_exc += -distance_exc;
				
				if (bprint) {printf("\n\nswap_src_N_s_start_exc=%d, distance_exc=%d\n", swap_src_N_s_start_exc, distance_exc);}

			}
		}	

		if ((swap_src_G_count_inh)|| (swap_src_G_count_exc > 0)){

			curandState local_state = randstates[neuron_idx];

			if (swap_src_G_count_inh > 0){
				generate_synapses(
					N, n,
					neuron_idx, N_rep,
					G_swap_tensor,
					swap_src_N_s_start_inh, swap_snk_N_s_start_inh,
					swap_src_G_count_inh, swap_snk_G_count_inh,
					group_neuron_counts_inh[snk_group_index],
					local_state,
					G_swap_tensor_shape_1,
					1,
					G_neuron_typed_ccount[swap_snk_G],
					0,
					bprint
				);
			}

			if (swap_src_G_count_exc > 0){
				generate_synapses(
					N, n,
					neuron_idx, N_rep,
					G_swap_tensor,
					swap_src_N_s_start_exc, swap_snk_N_s_start_exc,
					swap_src_G_count_exc, swap_snk_G_count_exc,
					group_neuron_counts_exc[snk_group_index],
					local_state,
					G_swap_tensor_shape_1,
					2,
					G_neuron_typed_ccount[G + swap_snk_G],
					group_neuron_counts_inh[snk_group_index],
					bprint
				);
			}

			randstates[neuron_idx] = local_state;
		}

		bool count = true;

		if (count){

			int swap_src_G_count = 0;
			int swap_snk_G_count = 0;
		
			for (int s=0; s < S; s++){
				snk_G = N_G[N_rep[n + s * N] * 2  + 1]; 
				G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = snk_G;
				swap_src_G_count += (snk_G == swap_src_G);
				swap_snk_G_count += (snk_G == swap_snk_G);
			}	
			neuron_group_counts[neuron_idx] = swap_src_G_count;
			neuron_group_counts[neuron_idx + G_swap_tensor_shape_1] = swap_snk_G_count;
		}
	
	}
}


void SnnSimulation::swap_groups(
	long* neurons, const int n_neurons, 
	long* groups, const int n_groups, 
	int* neuron_group_indices,
	int* G_swap_tensor, const int max_neurons_per_group, const int G_swap_tensor_shape_1,
	float* swap_rates,
	int* group_neuron_counts_inh, int* group_neuron_counts_exc, int* group_neuron_counts_total,
	int swap_delay,
	int* N_relative_G_indices, int* G_neuron_typed_ccount,
	int* neuron_group_counts,
	const int print_idx
)
{
	LaunchParameters lp_swap_groups = LaunchParameters(n_neurons, (void *)swap_groups_);

	//printf("\nswap groups %d, %d\n", n_groups, n_neurons);

	swap_groups_ KERNEL_ARGS2(lp_swap_groups.grid3, lp_swap_groups.block3)(
		neurons, n_neurons,
		groups, n_groups,
		neuron_group_indices,
		G_swap_tensor, max_neurons_per_group, G_swap_tensor_shape_1,
		swap_rates,
		group_neuron_counts_inh, group_neuron_counts_exc, group_neuron_counts_total,
		swap_delay,
		N_relative_G_indices, G_neuron_typed_ccount,
		N,
		G,
		S,
		N_G,
		N_rep,
		N_delays,
		rand_states,
		neuron_group_counts,
		print_idx
	);

	checkCudaErrors(cudaDeviceSynchronize());
}

void SnnSimulation::swap_groups_python(
	long neurons, const int n_neurons, 
	long groups, const int n_groups, 
	const long neuron_group_indices,
	const long G_swap_tensor, const int max_neurons_per_group, const int G_swap_tensor_shape_1,
	const long swap_rates,
	const long group_neuron_counts_inh, const long group_neuron_counts_exc, const long group_neuron_counts_total,
	const int swap_delay,
	const long N_relative_G_indices, const long G_neuron_typed_ccount,
	long neuron_group_counts,
	const int print_idx
)
{
	swap_groups(reinterpret_cast<long*> (neurons), n_neurons, 
				reinterpret_cast<long*> (groups), n_groups, 
				reinterpret_cast<int*> (neuron_group_indices),
				reinterpret_cast<int*> (G_swap_tensor), max_neurons_per_group, G_swap_tensor_shape_1,
				reinterpret_cast<float*> (swap_rates),
				reinterpret_cast<int*> (group_neuron_counts_inh), reinterpret_cast<int*> (group_neuron_counts_exc), reinterpret_cast<int*> (group_neuron_counts_total),
				swap_delay,
				reinterpret_cast<int*> (N_relative_G_indices), reinterpret_cast<int*> (G_neuron_typed_ccount),
				reinterpret_cast<int*> (neuron_group_counts),
				print_idx
				
	);
}
