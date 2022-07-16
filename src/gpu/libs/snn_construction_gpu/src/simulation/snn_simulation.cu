#include <simulation/snn_simulation.cuh>


__global__ void update_N_state_(
	const int N, 
	const int G,
	const float t,
	curandState* randstate, 
	float* N_pos,
	const int* N_G,
	const int* G_flags,
	const float* G_props,
	float* N_states,
	float* fired,
	int* last_fired,
	int* G_firing_count_hist, 
	const int t_mod_scatter_plot_length,
	const int row_sensory_input_type = 0,
	const int row_b_thalamic_input = 1,
	const int row_b_sensory_input = 3,
	const int row_b_monitor_group_firing_count = 6,
	const int row_thalamic_inh_input_current = 0,
	const int row_thalamic_exc_input_current = 1,
	const int row_sensory_input_current0 = 2,
	const int row_sensory_input_current1 = 3,
	bool b_monitor_group_firing_counts = true
)
{
	const int n = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (n < N)
	{
		curandState local_state = randstate[n];
		fired[n] = 0.f;
		N_pos[n * 13 + 10] = .3f;

		const int ntype = N_G[n * 2] - 1;
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

		if ((G_flags[src_G + row_b_thalamic_input * G] == 1) && (pt > 0.f) && (curand_uniform(&local_state) < pt))
		{
			const float rt = curand_uniform(&local_state);
			i += (G_props[src_G + row_thalamic_exc_input_current * G] * ntype 
				+ G_props[src_G + row_thalamic_inh_input_current * G] * (1 - ntype)) * rt;
		}
		
		if (G_flags[src_G + row_b_sensory_input * G] == 1)
		{
			const int input_type = G_flags[src_G + row_sensory_input_type * G];	
			if (input_type >= 0){
				i += (G_props[src_G + row_sensory_input_current1 * G] * input_type 
				      + G_props[src_G + row_sensory_input_current0 * G] * (1 - input_type));
			}
		}

		if (v > 30.f)
		{
			v = c;
			u = u + d;
			fired[n] = t;
			last_fired[n] = __float2int_rn(t);
			N_pos[n * 13 + 10] = 1.f;

			if ((b_monitor_group_firing_counts) && (ntype == 1) &&
			    (G_flags[src_G + row_b_monitor_group_firing_count * G] == 1)){
				
				atomicAdd(&G_firing_count_hist[src_G + t_mod_scatter_plot_length * G], 1);
				// printf("\nG_firing_count_hist[%d]=%d", src_G + t_mod_scatter_plot_length * G, 
				// 	G_firing_count_hist[src_G + t_mod_scatter_plot_length * G]);

			}			
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
	int* G_group_delay_counts_,
    int* G_flags_, 
    float* G_props_, 
    int* N_rep_, 
    int* N_rep_buffer_, 
    int* N_rep_pre_synaptic_idx_, 
    int* N_rep_pre_synaptic_counts_, 
    int* N_delays_, 
    float* N_states_,
	float* N_weights_,
	float* fired_,
	int* last_fired_,
	float* firing_times_,
	int* firing_idcs_,
	int* firing_counts_,
	int* G_firing_count_hist_,
	int* G_stdp_config0_,
	int* G_stdp_config1_,
	float* G_avg_weight_inh_,
	float* G_avg_weight_exc_,
	int* G_syn_count_inh_,
	int* G_syn_count_exc_
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
	G_group_delay_counts = G_group_delay_counts_;
    G_flags = G_flags_; 
    G_props = G_props_; 
    N_rep = N_rep_;

	N_rep_buffer = N_rep_buffer_;
    N_rep_pre_synaptic_idx = N_rep_pre_synaptic_idx_; 
    N_rep_pre_synaptic_counts = N_rep_pre_synaptic_counts_;

    N_delays = N_delays_;
    N_states = N_states_;
	N_weights = N_weights_;

	fired = fired_;
	last_fired = last_fired_;
	
	firing_times = firing_times_;
	firing_idcs = firing_idcs_;
	firing_counts = firing_counts_;
	G_firing_count_hist = G_firing_count_hist_;

	firing_times_write = firing_times;
	firing_times_read = firing_times;

	firing_idcs_write = firing_idcs;
	firing_idcs_read = firing_idcs;
	
	firing_counts_write = firing_counts;

	G_stdp_config0 = G_stdp_config0_;
	G_stdp_config1 = G_stdp_config1_;

	G_avg_weight_inh = G_avg_weight_inh_;
	G_avg_weight_exc = G_avg_weight_exc_;
	G_syn_count_inh = G_syn_count_inh_;
	G_syn_count_exc = G_syn_count_exc_;

	reset_firing_times_ptr_threshold = 13 * N;
	reset_firing_count_idx_threshold = 2 * T;

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
	const int* G_flags,
	const float* G_props,
	const int* N_rep, 
	// const int* Buffer,
	const int* N_rep_pre_synaptic_idx,
	const int* N_rep_pre_synaptic_counts,
	float* N_weights, 
	float* N_states,
	const int n_fired_m1_to_end,
	const int n_fired,
	const int t, 
	const int* N_delays,
	bool r_stdp,
	const int* G_stdp_config_current,
	const int* last_fired, 
	float alpha = 1.f,
	float beta = 0.f, 
	float phi_r = 1.f,
	float phi_p = 1.f,
	float a_r_p = .95f,
	float a_p_m = -.95f,
	float a_r_m = -.95f,
	float a_p_p = .95f,
	const int row_b_sensory_input = 3
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
		
		// bool bprint = fired_idx < 5;

		if (fired_idx < n_fired_m1_to_end)
		{
			// global index of firing-array < len(fired-array) 
			// -> use the trailing pointer
			n = fired_idcs_read[fired_idx];
			firing_time = __float2int_rn(firing_times_read[fired_idx]);
		}
		else
		{
			// global index of firing-array >= len(fired-array) 
			// -> use the 'normal' pointer
			n = fired_idcs[fired_idx - n_fired_m1_to_end];
			firing_time = __float2int_rn(firing_times[fired_idx - n_fired_m1_to_end]);
		}

		int delay = t - firing_time;
		// const int firing_time = __int2float_rn(firing_times[fired_idx]);

		// const int delay_idx = n * (D + 1) + (t - firing_time);

		// N_delays.shape = (D+1, N)
		// t - firing-time = delay -> use the delay to infer which synapses must be activated using
		// the N_delays array. 
		const int delay_idx = n + N * (delay);
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
		
		int src_G;
		
		if (r_stdp){
			src_G = N_G[n * 2 + 1];
		}

		int snk_N;
		int snk_G;
		// const int src_G = N_G[n * 2 + 1];
		bool is_sensory = false;
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
		int s_end = N_delays[delay_idx + N]; 
		int s_end2; 
		
		// float weight_delta = 0.f;

	
		float w;
		for (int s = N_delays[delay_idx]; s < s_end; s++)
		{
			idx = n + N * s;
			snk_N = N_rep[idx];
			snk_G = N_G[snk_N * 2 + 1];
			is_sensory = G_flags[snk_G + row_b_sensory_input * G] == 1;

			w  =  N_weights[idx];
			if (!is_sensory)
			{
				atomicAdd(&N_states[snk_N + 7 * N], w);
				
				
				if (r_stdp){

					int stdp_config = G_stdp_config_current[src_G + snk_G * G];
					if (((t - last_fired[snk_N]) < (delay)) 
						&& (stdp_config != 0)){

						w = fabsf(w);

						if ((w < .98) && (w > 0.02)){
							
							if (N_G[ 2 * snk_N] == 2){


								N_weights[idx] += ((stdp_config > 0) * phi_r * a_r_m + (stdp_config < 0) * phi_p * a_p_p) * w * (1. - w);

								if (false){
									printf("\nn=%d (t: %d), g=%d, sink=%d [%d](last fired: %d), w=%f (+%f), delay=%d",
										n, t, src_G, snk_N, snk_G, last_fired[snk_N], w, 
										(alpha * phi_r * a_r_m + beta * phi_p * a_p_p) * w * (1. - w), 
										delay);
								}
							} else {
								N_weights[idx] += ((stdp_config > 0) * phi_r * a_r_p + (stdp_config < 0) * phi_p * a_p_m) * w * (1. - w);
							}

						}
					} 
				}

			}
		}

		if (r_stdp && (delay == 0) && (!(G_flags[src_G + row_b_sensory_input * G] == 1))){
			int pre_src_N;
			float w2;
			s_end2 = N_rep_pre_synaptic_counts[n + 1];
			for (int s2 = N_rep_pre_synaptic_counts[n]; s2 < s_end2; s2++){

				idx = N_rep_pre_synaptic_idx[s2];

				w2 = fabsf(N_weights[idx]);
				if ((w2 < .98) && (w2 > 0.02))
				{				
					pre_src_N = idx - N * __float2int_rd(__int2float_rn(idx) / __int2float_rn(N));

					int stdp_config = G_stdp_config_current[N_G[pre_src_N * 2 + 1] + src_G * G];

					if (((t - last_fired[pre_src_N]) < (2 * D)) 
						&& (stdp_config != 0)){
							
						
						if (N_G[2 * n] == 2){
							N_weights[idx] += ((stdp_config > 0) * phi_r * a_r_p + (stdp_config < 0) * phi_p * a_p_m) * w2 * (1. - w2);
							if (false){
								printf("\nn=%d (t: %d) g=%d, idx=%d, pre-synaptic=%d [%d] (last fired: %d), w=%f (+%f)",
									n, t, src_G,idx, pre_src_N, N_G[pre_src_N * 2 + 1],last_fired[pre_src_N], w2, 
									(alpha * phi_r * a_r_p + beta * phi_p * a_p_m) * w2 * (1. - w2));
							}
						} else {
							N_weights[idx] += ((stdp_config > 0) * phi_r * a_r_m + (stdp_config < 0) * phi_p * a_p_p) * w2 * (1. - w2);
						}
					
					}
				}
			}
		}
	}
	
}


void SnnSimulation::update_plots()
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

void SnnSimulation::print_info(bool bprint_idcs, bool bprint_firing_times){
	std::cout << "\n\n  ------------------------------------ ";
	printf("\nt=%d", t);
	printf("\nn_fired=%d", n_fired);
	printf("\nn_fired_m1_to_end=%d", n_fired_m1_to_end);
	printf("\nn_fired_0=%d", n_fired_0);
	printf("\nn_fired_m1=%d", n_fired_m1);
	printf("\nn_fired_total=%d", n_fired_total);
	printf("\nn_fired_total_m1=%d", n_fired_total_m1);
	// printf("\nfiring_counts_write=%p", (void * )firing_counts_write);
	printf("\nfiring_counts_write=%ld", firing_counts_write - firing_counts);
	printf("\n\nfiring_idcs_read=%ld", firing_idcs_read - firing_idcs);
	printf("\nfiring_idcs_write=%ld", firing_idcs_write - firing_idcs);
	printf("\n\nfiring_times_read=%ld", firing_times_read - firing_times);
	printf("\nfiring_times_write=%ld", firing_times_write - firing_times);
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
	if (bprint_firing_times){
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
	}
	printf("\n");


}

void SnnSimulation::_update_sim_pointers(){

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
		// firing_counts_write = firing_counts;
		// firing_counts_idx = 1;
		n_fired_total = 0;
		// printf("\nt: %d (reset)\n", t);
		resetting = true;
	}

	printf("\n %d %d (%d/ %d)", n_fired_0, n_fired_m1, firing_counts_idx, reset_firing_count_idx_threshold);

	if (firing_counts_idx > reset_firing_count_idx_threshold){
		// printf("\nxxxxxxxxxxxxxxxxxx");
		firing_counts_idx = 1;
		firing_counts_write = firing_counts;
	} 
	
	if (firing_counts_idx_m1 > reset_firing_count_idx_threshold){
		// printf("\nyyyyyyyyyyyyyyyyyyyyyy");
		firing_counts_idx_m1 = 1;	
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
		// firing_counts_idx_m1 = 1;
		n_fired_m1_to_end = n_fired_total;
		n_fired_total_m1 = 0;
		// printf("\nt: %d (m1-reset)\n", t);
		resetting = false;
		// print_info();
	}
}

void SnnSimulation::update(const bool verbose)
{	
	t0 = std::chrono::steady_clock::now();

	// renderer->neurons_bodies.pos_colors.map_buffer();

	update_N_state_ KERNEL_ARGS2(lp_update_state.grid3, lp_update_state.block3 )(
		N,
		G,
		static_cast<float>(t),
		rand_states,
		N_pos,
		N_G,
		G_flags,
		G_props,
		N_states,
		fired,
		last_fired,
		G_firing_count_hist,
		t % scatter_plot_length
		// debug_i.get_dp(),
		// debug_v.get_dp()
    );

	// renderer->neurons_bodies.pos_colors.unmap_buffer();

	if (verbose)
	{
	 	std::cout << "\nt = " << t;
	}

	if (b_update_voltage_plot)
	{
		update_plots();
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


	checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(
		&n_fired_0, firing_counts + firing_counts_idx, sizeof(int), cudaMemcpyDeviceToHost));
	
	if (verbose) print_info(false, false);
	
	_update_sim_pointers();

	
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
		G_flags,
		G_props,
		N_rep,
		// N_rep_buffer,
		N_rep_pre_synaptic_idx,
		N_rep_pre_synaptic_counts,
		N_weights,
		N_states,
		n_fired_m1_to_end,
		n_fired,
		t,
		N_delays,
		stdp_active && (t > 100),
		G_stdp_config_current,
		last_fired
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

	update_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::steady_clock::now() - t0).count();
}


__device__ void roll_copy(
	
	int* write_array, int* read_array, 
	int write_col, int read_col, 
	int write_row_start, 
	int n_write_array_cols, int n_read_array_cols,
	const int copy_length, 
	const int read_offset, 
	bool bprint){
	
	int write_idx;
	int read_idx;
	//int roll_mod = abs(swap_snk_N_s_start - swap_src_N_s_start);

	for (int s=0; s < abs(copy_length); s++){

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
	int shift_start_offset,
	int* array0, int* array1,
	int col0, int col1,
	int n_cols0, int n_cols1,
	int end_row,
	int swap_dir,
	bool bprint
){
	int idx_end0 = col0 + (end_row) * n_cols0;
	int idx_end1 = col1 + (end_row) * n_cols1;
	int value0;

	if (swap_dir == 1){
		for (int k=shift_start_offset; k < 0; k++){
			value0 = array0[idx_end0 + (k+swap_dir) * n_cols0];
			if (value0 > 0){
	
				if (bprint){
					printf("\nN_rep[%d, %d] = %d -> %d, G_swap_index[%d, %d] = %d -> %d",
						end_row + k, idx_end0, array0[idx_end0 + k * n_cols0], value0,
						end_row + k, idx_end1, array1[idx_end1 + k * n_cols1], array1[idx_end1 + (k+swap_dir) * n_cols1]
					);
		
				}
	
				array0[idx_end0 + k * n_cols0] = value0;
				array1[idx_end1 + k * n_cols1] = array1[idx_end1 + (k+swap_dir) * n_cols1];
			}
		
		}
	}
	
	else if (swap_dir == -1){
		for (int k=0; k > shift_start_offset; k--){
			value0 = array0[idx_end0 + (k+swap_dir) * n_cols0];
			if (value0 > 0){
	
				if (bprint){
					printf("\nN_rep[%d, %d] = %d -> %d, G_swap_index[%d, %d] = %d -> %d",
						end_row + k, idx_end0, array0[idx_end0 + k * n_cols0], value0,
						end_row + k, idx_end1, array1[idx_end1 + k * n_cols1], array1[idx_end1 + (k+swap_dir) * n_cols1]
					);
		
				}
	
				array0[idx_end0 + k * n_cols0] = value0;
				array1[idx_end1 + k * n_cols1] = array1[idx_end1 + (k+swap_dir) * n_cols1];
			}
		
		}
	}

}


__device__ void generate_synapses(
	const int N,
	const int n,
	const int neuron_idx,
	int* N_rep,
	int* G_swap_tensor,
	int& swap_src_N_s_start, int& swap_snk_N_s_start,
	int& swap_src_G_count, int& swap_snk_G_count,
	const int max_snk_count,
	curandState &local_state,
	int G_swap_tensor_shape_1, 
	const int swap_type,
	const int index_offset,
	const int relative_index_offset,
	const int swap_dir,
	bool bprint
){
	
	int snk_N;
	int min_G_swap_snk = G_swap_tensor[neuron_idx + swap_snk_N_s_start * G_swap_tensor_shape_1];
	int max_G_swap_snk = G_swap_tensor[neuron_idx + (swap_snk_N_s_start + swap_snk_G_count - 1) * G_swap_tensor_shape_1];
	if (swap_snk_G_count == 0){
		min_G_swap_snk = max_snk_count + relative_index_offset;
		max_G_swap_snk = -1;
	}
	float r;

	int s_end = swap_src_N_s_start + swap_src_G_count;
	

	for (int s=swap_src_N_s_start; s < s_end; s++){
	// for (int s=swap_src_N_s_start; s < swap_src_N_s_start + 2; s++){
		
		r = curand_uniform(&local_state);

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

			int last_write_mode = 0;
			int write_mode = 0;

			// while ((!found) && (j < 40)){
			while ((!found) && (j < G_swap_tensor_shape_1)){
				
				write_mode = 0;
				//write_row = s - s_offset;
				// write = -i;
				swap_idx = neuron_idx + (swap_snk_N_s_start + i )  * G_swap_tensor_shape_1;
				
				G_swap0 = G_swap_tensor[swap_idx];
				G_swap_m1 = G_swap_tensor[swap_idx - G_swap_tensor_shape_1];


				if((snk_N < min_G_swap_snk) || (swap_snk_G_count == 0)){
				
	
					min_G_swap_snk = snk_N;
					
					if (swap_dir == 1){
						write_row = swap_snk_N_s_start - 1;
					} else {
						write_row = swap_snk_N_s_start;
					}
					write_mode = 1;

					if (swap_snk_G_count == 0){
						max_G_swap_snk = snk_N;
					}
					// G_swap_tensor[swap_idx - G_swap_tensor_shape_1] = G_swap0;	
					// G_swap_tensor[swap_idx] = snk_N;		
				}
				else if((snk_N > max_G_swap_snk)){
					write_mode = 2;
					// if (swap_snk_G_count == 0){
					// 	min_G_swap_snk = snk_N;
					// }
					max_G_swap_snk = snk_N;
					if (swap_dir == 1){
						write_row = swap_snk_N_s_start + swap_snk_G_count - 1;
					} else {
						write_row = swap_snk_N_s_start + swap_snk_G_count;
					}
					

				}
				else if ((G_swap_m1 < snk_N) && (snk_N < G_swap0)){
					write_mode = 3;
					if (swap_dir == 1){
						write_row = swap_snk_N_s_start + i - 1;
					} else {
						write_row = swap_snk_N_s_start + i;
					}
					
				}

				found = write_mode > 0;

				if (found){
					if (swap_dir == 1){
						swap_snk_N_s_start -= 1;
					} else {
						swap_src_N_s_start += 1;
					}
					
					// write = snk_N;
					// s_offset++;
					swap_snk_G_count++;
					swap_src_G_count--;
					// G_swap_tensor[neuron_idx + (write_row) * G_swap_tensor_shape_1] = write;
					break;}
				

				if ((snk_N == G_swap0)){
					snk_N = (snk_N + 1) % max_snk_count;
				}

				// if (bprint || (j >= 30)) {
				if (bprint) {
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

			// if (bprint || (j >= 30)) {
			if (false) {
				printf("\n[%d, %d] (found j=%d, mod:%d->%d) N_rep[%d, %d]=%d (%d) [%d (snk_N) + %d - %d]", 
					n, neuron_idx, j, last_write_mode, write_mode,
					write_row, n, N_rep[n + (write_row) * N], N_rep[n + (swap_snk_N_s_start-1) * N], snk_N,
					index_offset, relative_index_offset);
			}

			//|| (j >= 30)
			if ((swap_dir > 0) && (write_mode > 1)){
				shift_values_row_wise_(
					swap_snk_N_s_start - write_row - 1,
					N_rep, G_swap_tensor,
					n, neuron_idx,
					N, G_swap_tensor_shape_1,
					write_row,
					swap_dir,
					bprint
				);
			} else if ((swap_dir < 0) && (write_mode > 0) && (write_mode != 2)){
				shift_values_row_wise_(
					write_row - swap_src_N_s_start - 1,
					N_rep, G_swap_tensor,
					n, neuron_idx,
					N, G_swap_tensor_shape_1,
					swap_src_N_s_start,
					swap_dir,
					bprint
				);
			}


			N_rep[n + (write_row) * N] = snk_N + index_offset - relative_index_offset;
			G_swap_tensor[neuron_idx + (write_row) * G_swap_tensor_shape_1] = snk_N;

			// bprint || (j >= 30)

			if (bprint  ) {
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
	const int* neuron_group_indices,
	int* G_swap_tensor, const int G_swap_tensor_shape_1,
	const float* swap_rates,
	const int* group_neuron_counts_inh, const int* group_neuron_counts_exc, const int* group_neuron_counts_total, 
	const int* G_delay_distance,
	const int* N_relative_G_indices, const int* G_neuron_typed_ccount,
	int N,
	int G,
	int S,
	int D,
	int* N_G,
	int* N_rep,
	int* N_delays,
	curandState* randstates,
	int* neuron_group_counts,
	const int expected_snk_type,
	const int print_idx
){
	const int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x; 

	if (neuron_idx < n_neurons){

		// bool bprint = (neuron_idx == min(print_idx, n_neurons- 1));
		bool bprint = false;

		const int n = neurons[neuron_idx];
		
		const int group_index = neuron_group_indices[neuron_idx];
		const int snk_group_index = group_index + 2 * n_groups;

		const int swap_src_G = groups[group_index];
		const int src_G = groups[snk_group_index - n_groups];
		const int swap_snk_G = groups[snk_group_index];

		const float swap_rate = swap_rates[group_index];

		const int total_src_G_count = group_neuron_counts_total[group_index];
		const int total_snk_G_count = group_neuron_counts_total[snk_group_index];

		if (bprint){		
			printf("\n\nswap_src %d (%d), src_G %d %d (%d), swap_snk %d (%d)  neuron_group_indices[%d] = %d\n", 
			swap_src_G, total_src_G_count,
			N_G[2 * n + 1], src_G, group_neuron_counts_total[snk_group_index - n_groups],
			swap_snk_G, total_snk_G_count, neuron_idx, (int)neuron_group_indices[neuron_idx]);
		}

		int snk_N;
		int snk_type;
		int snk_G;

		int swap_delay_src = G_delay_distance[swap_src_G + src_G * G];
		int swap_delay_snk = G_delay_distance[swap_snk_G + src_G * G];

		int s_start = N_delays[n + min(swap_delay_src, swap_delay_snk) * N];
		int s_end =  N_delays[n + (max(swap_delay_src, swap_delay_snk) + 1) * N];

		int swap_src_N_s_start = s_start;
		int swap_snk_N_s_start = s_start;

		int swap_src_G_count = 0;
		int swap_snk_G_count = 0;

		for (int s=s_start; s < s_end; s++)
		{
			
			snk_N = N_rep[n + s * N];
			snk_type = N_G[snk_N * 2];
			

			if (snk_type == expected_snk_type){
				
				
				snk_G = N_G[snk_N * 2 + 1];

				if (snk_G == swap_src_G)
				{
					
					if (swap_src_G_count == 0){
						swap_src_N_s_start = s;
					}
					swap_src_G_count += 1;

					G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = -total_src_G_count-N_rep[n + s * N]; //-2;
					if (bprint) printf("\n(%d) n_snk=%d, (snk_G=%d)  (s=%d) %d %d, src_counts=[%d, ]", 
						n, N_rep[n + s * N], snk_G, s, snk_G == swap_src_G, snk_G == swap_snk_G, 
						swap_src_G_count);
					
					N_rep[n + s * N] = -1;
					
				}
				else if (snk_G == swap_snk_G)
				{
					if (snk_type == expected_snk_type){
						if (swap_snk_G_count == 0){
							swap_snk_N_s_start = s;
						}
						swap_snk_G_count += 1;
					} 

					G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = N_relative_G_indices[snk_N];
					
					if (bprint) printf("\n(%d) n_snk=%d, (snk_G=%d)  (s=%d) %d %d, snk_N_rel=%d", 
						n, N_rep[n + s * N], snk_G, s, 
						snk_G == swap_src_G, snk_G == swap_snk_G, 
						N_relative_G_indices[snk_N]);
				} 
				else if((swap_src_G_count > 0) || (swap_snk_G_count > 0)){
					G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = N_rep[n + s * N];	
				}
			} 
			else if((swap_src_G_count > 0) || (swap_snk_G_count > 0))
			{
				G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = N_rep[n + s * N];	
			}
			
		}

		if (swap_snk_G_count == 0){
			swap_snk_N_s_start = swap_src_N_s_start + swap_src_G_count;
		}
		// if (swap_snk_G_count_exc == 0){
		// 	swap_snk_N_s_start_exc += 1;
		// }

		if (swap_rate < 1.f){
			s_end = swap_src_N_s_start + swap_src_G_count;

			swap_src_G_count = __float2int_rd (__int2float_rn(swap_src_G_count) * swap_rate);

			for (int s=swap_src_N_s_start + swap_src_G_count; s < s_end; s++)
			{
				snk_N = - G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] - total_src_G_count;
				G_swap_tensor[neuron_idx + s * G_swap_tensor_shape_1] = snk_N;
				N_rep[n + s * N] =  snk_N;
			}
		}


		if (bprint){
			printf("\n\nrow intervals: src=[%d, %d (+%d)) snk=[%d, %d (+%d)), swap_rate=%f\n", 
				   swap_src_N_s_start, swap_src_N_s_start + swap_src_G_count, 
				   swap_src_G_count, 
				   swap_snk_N_s_start, swap_snk_N_s_start + swap_snk_G_count,
				   swap_snk_G_count, swap_rate);
					//    printf("exc: src=[%d, +%d] snk=[%d, +%d]\n", 
					//    swap_src_N_s_start_exc, swap_src_G_count_exc, swap_snk_N_s_start_exc, swap_snk_G_count_exc);
		}

		if (swap_src_G_count > 0){

			int distance = max(swap_snk_N_s_start - (swap_src_N_s_start + swap_src_G_count), 
						       min(0, swap_snk_N_s_start + swap_snk_G_count - swap_src_N_s_start));

			int swap_dir = 1 * (swap_snk_N_s_start > swap_src_N_s_start) + -1 * (swap_snk_N_s_start < swap_src_N_s_start);
			
			if (distance != 0){

				// if (swap_dir == 1){
					roll_copy(
						N_rep, G_swap_tensor, 
						n, neuron_idx, 
						min(swap_src_N_s_start, swap_snk_N_s_start + swap_snk_G_count), 
						N, G_swap_tensor_shape_1, 
						(swap_snk_N_s_start - swap_src_N_s_start) * (swap_dir == 1) + (swap_dir == -1) * (-distance + swap_src_G_count), 
						swap_src_G_count * (swap_dir == 1) - distance *  (swap_dir == -1), 
						bprint);
				// } else {
				// 	roll_copy(
				// 		N_rep, G_swap_tensor, 
				// 		n, neuron_idx, 
				// 		swap_snk_N_s_start, 
				// 		N, G_swap_tensor_shape_1, 
				// 		swap_snk_N_s_start - swap_src_N_s_start, 
				// 		swap_dir * swap_src_G_count, 
				// 		bprint);
				// }


				swap_src_N_s_start += distance;

				if (bprint) {printf("\n\nswap_src_N_s_start=%d, distance=%d\n", swap_src_N_s_start, distance);}

			}


			curandState local_state = randstates[neuron_idx];

			int max_snk_count;
			int index_offset;
			int relative_index_offset;
			if (expected_snk_type == 1){
				max_snk_count = group_neuron_counts_inh[snk_group_index];
				index_offset = G_neuron_typed_ccount[swap_snk_G];
				relative_index_offset = 0;
			}
			else if (expected_snk_type == 2){
				max_snk_count = group_neuron_counts_exc[snk_group_index];
				index_offset = G_neuron_typed_ccount[G + swap_snk_G];
				relative_index_offset = group_neuron_counts_inh[snk_group_index];
			}

			if (swap_src_G_count > 0){
				generate_synapses(
					N, n,
					neuron_idx, N_rep,
					G_swap_tensor,
					swap_src_N_s_start, swap_snk_N_s_start,
					swap_src_G_count, swap_snk_G_count,
					max_snk_count,
					local_state,
					G_swap_tensor_shape_1,
					expected_snk_type,
					index_offset,
					relative_index_offset,
					swap_dir,
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

			if (swap_delay_src != swap_delay_snk){
				
				int d1 = max(swap_delay_src, swap_delay_snk);
				int count0 = swap_delay_src;

				for (int d=min(swap_delay_src, swap_delay_snk); d < d1; d++){
					N_delays[n + d * N] -= count0;
				}
			}
		}
	
	}
}


void SnnSimulation::swap_groups(
	long* neurons, const int n_neurons, 
	long* groups, const int n_groups, 
	int* neuron_group_indices,
	int* G_swap_tensor, const int G_swap_tensor_shape_1,
	float* swap_rates_inh, float* swap_rates_exc,
	int* group_neuron_counts_inh, int* group_neuron_counts_exc, int* group_neuron_counts_total,
	int* G_delay_distance,
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
		G_swap_tensor, G_swap_tensor_shape_1,
		swap_rates_inh,
		group_neuron_counts_inh, group_neuron_counts_exc, group_neuron_counts_total,
		G_delay_distance,
		N_relative_G_indices, G_neuron_typed_ccount,
		N,
		G,
		S,
		D,
		N_G,
		N_rep,
		N_delays,
		rand_states,
		neuron_group_counts,
		1,
		print_idx
	);

	checkCudaErrors(cudaDeviceSynchronize());

	swap_groups_ KERNEL_ARGS2(lp_swap_groups.grid3, lp_swap_groups.block3)(
		neurons, n_neurons,
		groups, n_groups,
		neuron_group_indices,
		G_swap_tensor, G_swap_tensor_shape_1,
		swap_rates_exc,
		group_neuron_counts_inh, group_neuron_counts_exc, group_neuron_counts_total,
		G_delay_distance,
		N_relative_G_indices, G_neuron_typed_ccount,
		N,
		G,
		S,
		D,
		N_G,
		N_rep,
		N_delays,
		rand_states,
		neuron_group_counts,
		2,
		print_idx
	);

	checkCudaErrors(cudaDeviceSynchronize());
}

void SnnSimulation::swap_groups_python(
	long neurons, const int n_neurons, 
	long groups, const int n_groups, 
	const long neuron_group_indices,
	const long G_swap_tensor, const int G_swap_tensor_shape_1,
	const long swap_rates_inh, const long swap_rates_exc,
	const long group_neuron_counts_inh, const long group_neuron_counts_exc, const long group_neuron_counts_total,
	const long G_delay_distance, 
	const long N_relative_G_indices, const long G_neuron_typed_ccount,
	long neuron_group_counts,
	const int print_idx
)
{
	swap_groups(reinterpret_cast<long*> (neurons), n_neurons, 
				reinterpret_cast<long*> (groups), n_groups, 
				reinterpret_cast<int*> (neuron_group_indices),
				reinterpret_cast<int*> (G_swap_tensor), G_swap_tensor_shape_1,
				reinterpret_cast<float*> (swap_rates_inh), reinterpret_cast<float*> (swap_rates_exc),
				reinterpret_cast<int*> (group_neuron_counts_inh), reinterpret_cast<int*> (group_neuron_counts_exc), reinterpret_cast<int*> (group_neuron_counts_total),
				reinterpret_cast<int*> (G_delay_distance),
				reinterpret_cast<int*> (N_relative_G_indices), reinterpret_cast<int*> (G_neuron_typed_ccount),
				reinterpret_cast<int*> (neuron_group_counts),
				print_idx
				
	);
}

void SnnSimulation::set_stdp_config(int stdp_config_id, bool activate){

	if (stdp_config_id==0){
		G_stdp_config_current = G_stdp_config0;
	} else if (stdp_config_id==1){
		G_stdp_config_current = G_stdp_config1;
	} else {
		throw std::invalid_argument( "not in [0, 1]" );
	}

	if (activate){
		stdp_active = true;
	}
}


__global__ void reset_N_rep_pre_synaptic(
	const int N,
	const int S,
	int* Buffer,
	int* N_rep_pre_synaptic_idx,
	int* N_rep_pre_synaptic_counts
){
	const int src_N = blockIdx.x * blockDim.x + threadIdx.x; 
	if (src_N < N){
		for (int s = 0; s < S; s++){
			Buffer[src_N + s * N] = -1;
			N_rep_pre_synaptic_idx[src_N + s * N] = -1;
		}

		if (src_N == 0){
			N_rep_pre_synaptic_counts[0] = 0;
		}
		N_rep_pre_synaptic_counts[src_N + 1] = 0;

	}

}


__global__ void reset_N_rep_snk_counts(
	const int N,
	int* N_rep_pre_synaptic_counts
){
	const int src_N = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (src_N < N){

		if (src_N == 0){
			N_rep_pre_synaptic_counts[0] = 0;
		}

		N_rep_pre_synaptic_counts[src_N + 1] = 0;
	}
}


__global__ void fill_N_rep_snk_counts(
	const int N,
	const int S,
	int* N_rep,
	int* N_rep_pre_synaptic_counts
){
	const int src_N = blockIdx.x * blockDim.x + threadIdx.x; 
	int snk_N;
	
	if (src_N < N){

		for (int s = 0; s < S; s++){
			snk_N = N_rep[src_N + s * N];

			if (snk_N == -1){
				printf("\n %d", src_N);
			}

			atomicAdd(&N_rep_pre_synaptic_counts[snk_N + 1], 1);
		}
	}
}


__global__ void fill_N_rep_pre_synaptic_buffer(
	const int N,
	const int S,
	int* N_rep,
	int* Buffer,
	int* N_rep_pre_synaptic_idx,
	int* N_rep_pre_synaptic_counts
){
	
	// Fill Buffer with the indices of the sysnapses in N_rep.
	//  "N_rep-synapse-index" - int("N_rep-synapse-index" / N) * N yields the pre-Synaptic Neuron. 
	// The write indices (for the Buffer-array) are given by N_rep_pre_synaptic_counts.
	// The values in Buffer will be copied to N_rep_pre_synaptic_idx in the 
	// fill_N_rep_pre_synaptic_idx-kernel.

	const int src_N = blockIdx.x * blockDim.x + threadIdx.x; 

	if (src_N < N){

		int snk_N;
		int write_idx;
	
		int synapse_idx;

		for (int s = 0; s < S; s++){
			
			synapse_idx = src_N + s * N;

			snk_N = N_rep[synapse_idx];
			write_idx = N_rep_pre_synaptic_counts[snk_N];
			
			while (synapse_idx != -1){
				
				synapse_idx = atomicExch(&Buffer[write_idx], synapse_idx);
				N_rep_pre_synaptic_idx[write_idx] = snk_N;
				write_idx++;
			}

			atomicAdd(&N_rep_pre_synaptic_counts[snk_N],1);

		}
	}

}


void sort_N_rep_sysnaptic(
	const int N,
	const int S,
	int* sort_keys,
	int* Buffer,
	int* N_rep_pre_synaptic_counts,
	const bool verbose = true
){

	auto sort_keys_dp = thrust::device_pointer_cast(sort_keys);
	auto N_rep_dp = thrust::device_pointer_cast(Buffer);
	auto N_rep_counts_dp = thrust::device_pointer_cast(N_rep_pre_synaptic_counts);

	int n_sorted = 0;
	int N_batch_size = 50000;
	int S_batch_size;

	std::string msg;
	if (verbose) {
		msg = "sorted: 0/" + std::to_string(N);
		std::cout << msg;
	}

	while (n_sorted < N){
			
	 	if (n_sorted + N_batch_size > N){
	 		N_batch_size = N - n_sorted;
		} 

		// printf("\nN_batch_size=%d", N_batch_size);

		S_batch_size = N_rep_counts_dp[n_sorted + N_batch_size] - N_rep_counts_dp[n_sorted];

		// printf("\nS_batch_size=%d\n", S_batch_size);

	 	thrust::stable_sort_by_key(N_rep_dp, N_rep_dp + S_batch_size, sort_keys_dp);
	 	thrust::stable_sort_by_key(sort_keys_dp, sort_keys_dp + S_batch_size, N_rep_dp);
		
	 	n_sorted += N_batch_size;
	 	sort_keys_dp += S_batch_size;
	 	N_rep_dp += S_batch_size;

	 	if (verbose) { 
	 		std::cout << std::string(msg.length(),'\b');
	 		msg = "sorted: " + std::to_string(n_sorted) + "/" + std::to_string(N);
	 		std::cout << msg;
	 	}
	}

	if (verbose) printf("\n");

}


__device__ int v = 0;

__global__ void write_pre_synaptic_idx(
	int s,
	int src_N,
	int start,
	int end,
	const int* Buffer,
	int* N_rep_pre_synaptic_idx
) 
{
	const int write_idx = start + blockIdx.x * blockDim.x + threadIdx.x; 

	if ((write_idx < end) && (Buffer[write_idx] == src_N)){

		N_rep_pre_synaptic_idx[write_idx] = s;

	}
}



__global__ void fill_N_rep_pre_synaptic_idx(
	const int N,
	const int S,
	int* Buffer,
	int* N_rep_pre_synaptic_idx
){
	const int n = blockIdx.x * blockDim.x + threadIdx.x; 

	if (n < N){

		int snk_N;
		int idx;


		for (int j=0; j < S; j++){

			idx = n + j * N; 

			snk_N = N_rep_pre_synaptic_idx[idx];

			N_rep_pre_synaptic_idx[idx] = Buffer[idx]; //(Buffer[idx] - snk_N) / N;

			Buffer[idx] = snk_N;

		}
	}
}


void SnnSimulation::actualize_N_rep_pre_synaptic(){

	LaunchParameters launch_pars = LaunchParameters(N, (void *)reset_N_rep_pre_synaptic);

	checkCudaErrors(cudaDeviceSynchronize());

	reset_N_rep_pre_synaptic KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		S,
		N_rep_buffer,
		N_rep_pre_synaptic_idx,
		N_rep_pre_synaptic_counts
	);

	checkCudaErrors(cudaDeviceSynchronize());

	fill_N_rep_snk_counts KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		S,
		N_rep,
		N_rep_pre_synaptic_counts
	);

	thrust::device_ptr<int> count_dp = thrust::device_pointer_cast(N_rep_pre_synaptic_counts);

	checkCudaErrors(cudaDeviceSynchronize());

	thrust::inclusive_scan(thrust::device, count_dp, count_dp + N + 1, count_dp);

	checkCudaErrors(cudaDeviceSynchronize());

	fill_N_rep_pre_synaptic_buffer KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		S,
		N_rep,
		N_rep_buffer,
		N_rep_pre_synaptic_idx,
		N_rep_pre_synaptic_counts
	);

	checkCudaErrors(cudaDeviceSynchronize());

	reset_N_rep_snk_counts KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		N_rep_pre_synaptic_counts
	);	

	checkCudaErrors(cudaDeviceSynchronize());

	fill_N_rep_snk_counts KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N,
		S,
		N_rep,
		N_rep_pre_synaptic_counts
	);

	checkCudaErrors(cudaDeviceSynchronize());

	thrust::inclusive_scan(thrust::device, count_dp, count_dp + N + 1, count_dp);

	checkCudaErrors(cudaDeviceSynchronize());

	sort_N_rep_sysnaptic(N, S, N_rep_pre_synaptic_idx, N_rep_buffer, N_rep_pre_synaptic_counts);

	checkCudaErrors(cudaDeviceSynchronize());

	printf("\nfill_N_rep_pre_synaptic_idx...");

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	fill_N_rep_pre_synaptic_idx KERNEL_ARGS2(launch_pars.grid3, launch_pars.block3)(
		N, S, 
		N_rep_buffer, 
		N_rep_pre_synaptic_idx
	);
	
	checkCudaErrors(cudaDeviceSynchronize());

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	printf(" done.\n");

	//std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
}


__global__ void reset_G_avg_weight_G_syn_count_(
	const int G,
	float* G_avg_weight_inh,
	float* G_avg_weight_exc,
	int* G_syn_count_inh,
	int* G_syn_count_exc
){
	const int src_G = blockIdx.x * blockDim.x + threadIdx.x; 
	if (src_G < G){
		int write_idx;
		for (int snk_G = 0; snk_G < G; snk_G++){
			write_idx = src_G + snk_G * G;
			G_avg_weight_inh[write_idx] = 0.f;
			G_syn_count_inh[write_idx] = 0;
			G_avg_weight_exc[write_idx] = 0.f;
			G_syn_count_exc[write_idx] = 0;
		}
	}
}


__global__ void prefill_G_avg_weight_G_syn_count_(
	const int N,
	const int G,
	const int S,
	const int* N_G,
	const int* G_group_delay_counts,
	const int* N_rep,
	const float* N_weights,
	float* G_avg_weight_inh,
	float* G_avg_weight_exc,
	int* G_syn_count_inh,
	int* G_syn_count_exc
){
	const int src_N = blockIdx.x * blockDim.x + threadIdx.x; 

	if (src_N < N){
		

		int src_type = N_G[src_N * 2];
		int src_G = N_G[src_N * 2 + 1];
		int snk_N;
		int snk_G;
		int N_rep_idx;
		int write_idx;
		float weight;

		for (int s = 0; s < S; s++){
			
			N_rep_idx = src_N + s * N;
			snk_N = N_rep[N_rep_idx];
			snk_G = N_G[snk_N * 2 + 1];
			write_idx = src_G + snk_G * G;
			weight = N_weights[N_rep_idx];
			
			if (src_type == 1){
				atomicAdd(&G_avg_weight_inh[write_idx], 100 * weight);
				atomicAdd(&G_syn_count_inh[write_idx], 1);
			} else if (src_type == 2){
				atomicAdd(&G_avg_weight_exc[write_idx], 100 * weight);
				atomicAdd(&G_syn_count_exc[write_idx], 1);
			}
		}

	}
}

__global__ void fill_G_avg_weight_(
	const int G,
	float* G_avg_weight_inh,
	float* G_avg_weight_exc,
	const int* G_syn_count_inh,
	const int* G_syn_count_exc
){
	const int src_G = blockIdx.x * blockDim.x + threadIdx.x; 

	if (src_G < G){
		
		int count;
		int write_idx;

		for (int snk_G = 0; snk_G < G; snk_G++){

			write_idx = src_G + snk_G * G;

			count = G_syn_count_inh[write_idx];
			if (count != 0){
				G_avg_weight_inh[write_idx] /= 100.f * __int2float_rn(count);
			}

			count = G_syn_count_exc[write_idx];
			if (count != 0){
				G_avg_weight_exc[write_idx] /= 100.f * __int2float_rn(count);
			}
		}

	}
}


void SnnSimulation::calculate_avg_group_weight(){

	LaunchParameters launch_pars_N = LaunchParameters(N, (void *)prefill_G_avg_weight_G_syn_count_);
	LaunchParameters launch_pars_G = LaunchParameters(N, (void *)reset_G_avg_weight_G_syn_count_);

	checkCudaErrors(cudaDeviceSynchronize());

	reset_G_avg_weight_G_syn_count_ KERNEL_ARGS2(launch_pars_G.grid3, launch_pars_G.block3)(
		G,
		G_avg_weight_inh,
		G_avg_weight_exc,
		G_syn_count_inh,
		G_syn_count_exc
	);

	checkCudaErrors(cudaDeviceSynchronize());

	prefill_G_avg_weight_G_syn_count_ KERNEL_ARGS2(launch_pars_N.grid3, launch_pars_N.block3)(
		N,
		G,
		S,
		N_G,
		G_group_delay_counts,
		N_rep,
		N_weights,
		G_avg_weight_inh,
		G_avg_weight_exc,
		G_syn_count_inh,
		G_syn_count_exc
	);

	checkCudaErrors(cudaDeviceSynchronize());

	fill_G_avg_weight_ KERNEL_ARGS2(launch_pars_G.grid3, launch_pars_G.block3)(
		G,
		G_avg_weight_inh,
		G_avg_weight_exc,
		G_syn_count_inh,
		G_syn_count_exc
	);

	checkCudaErrors(cudaDeviceSynchronize());

}


void SnnSimulation::set_plotting_pointers(
	float* voltage_plot_data_,
	float* scatter_plot_data_
){
	voltage_plot_data = voltage_plot_data_;
	scatter_plot_data = scatter_plot_data_;
}
void SnnSimulation::set_plotting_pointers_python(
	const long voltage_plot_data_dp,
	const long scatter_plot_data_dp
){
	set_plotting_pointers(reinterpret_cast<float*> (voltage_plot_data_dp),
			     reinterpret_cast<float*> (scatter_plot_data_dp));
}