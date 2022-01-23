#include <snn_construction.cuh>


void __global__ fill_N_G_group_id_and_G_neuron_count_per_type_(
	const int N,
    const int G,
	const float* N_pos, 
	float N_pos_shape_x, float N_pos_shape_y, float N_pos_shape_z,
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
		const int x = __float2int_rd(((N_pos[n * N_pos_n_cols] / N_pos_shape_x) * min_G_shape));  
		const int y = __float2int_rd(((N_pos[n * N_pos_n_cols + 1] / N_pos_shape_y) * min_G_shape));
		const int z = __float2int_rd(((N_pos[n * N_pos_n_cols + 2] / N_pos_shape_z) * min_G_shape));

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
    const float* N_pos,
	const int N_pos_shape_x, const int N_pos_shape_y, const int N_pos_shape_z,
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
		static_cast<float>(N_pos_shape_x), static_cast<float>(N_pos_shape_y), static_cast<float>(N_pos_shape_z),
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
	printf("\n");
}


__device__ float sigmoidal_connection_probability(
	const float delay,
	const float max_delay,
	const float alpha = 1.f,
	const float beta = 1.f,
	const float gamma = 0.125f
)
{
	const float inv_max_delay = (1.f / max_delay);
	const float normalized_delay = delay * inv_max_delay;
	
	const float sigmoid = 0.5f - (1 / (1 + expf(-(alpha * delay - 1.f))));
	const float offset = inv_max_delay * (0.5f + gamma - gamma * powf(normalized_delay, 2.f));

	return fminf(1.f, beta * (inv_max_delay * sigmoid + offset));
}


__global__ void fill_G_neuron_count_per_delay_(
		const float fS,
		const int D,
		const float fD,
		const int G,
		const int* G_delay_distance,
		int* G_neuron_counts
)
{
	// connection probabilities from inh -> exc
	// connection probabilities from exc -> (inh & exc)

	const int g = threadIdx.x + blockDim.x * blockIdx.x;

	if (g < G)
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


void fill_G_neuron_count_per_delay(
	const int S,
	const int D,
	const int G,
	const int* G_delay_distance,
	int* G_neuron_counts
)
{	
	cudaDeviceSynchronize();
	LaunchParameters launch(G, (void *)fill_G_neuron_count_per_delay_); 

	fill_G_neuron_count_per_delay_ KERNEL_ARGS2(launch.grid3, launch.block3)(
		static_cast<float>(S),
		D,
		static_cast<float>(D),
		G,
		G_delay_distance,
		G_neuron_counts
	);
	
	cudaDeviceSynchronize();
	printf("\n");
}


__device__ void expected_syn_count(
	const float fD,
	const int D, 
	const int G, 
	const int* G_neuron_counts,
	const int ioffs_inh,
	const int ioffs_exc,

	const float alpha_inh, const float beta_inh, const float gamma_inh,
	const float alpha_exc, const float beta_exc, const float gamma_exc,
	
	float* exp_cnt_inh, float* exp_cnt_exc,
	const int group,
	const bool verbose = 1,
	const int print_group = 1
)
{
	*exp_cnt_inh = 0;
	*exp_cnt_exc = 0;

	
	for (int delay = 0; delay < D; delay++)
	{
		const int idx = (delay)*G;
		// # inh targets (exc)
		const float n_inh_targets = __int2float_rn(G_neuron_counts[ioffs_exc + idx]);
		// # exc targets (inh & exc)
		float n_exc_targets = n_inh_targets + __int2float_rn(G_neuron_counts[ioffs_inh + idx]);

		if ((delay == 0) && (G_neuron_counts[ioffs_inh - G] > 0))
		{
			// only exc neurons will have a technical probability > 0 to form an autapse
			n_exc_targets-= 1.f;
		}

		const float fdelay = __int2float_rn(delay);

		const float prob_inh = sigmoidal_connection_probability(fdelay, fD, alpha_inh, beta_inh, gamma_inh);
		const float prob_exc = sigmoidal_connection_probability(fdelay, fD, alpha_exc, beta_exc, gamma_exc);
		if (n_inh_targets > 0){
			*exp_cnt_inh += roundf(n_inh_targets * prob_inh + .5);
		}
		if (n_exc_targets > 0){
			*exp_cnt_exc += roundf(n_exc_targets * prob_exc + .5);
		}
		if ((verbose) && (group == print_group)){
			printf("\ninh expected_syncount = %f (++ %f)", *exp_cnt_inh, n_inh_targets * prob_inh);
			printf("\n(exc) a=%f,b=%f,g=%f", alpha_exc, beta_exc, gamma_exc);
			printf("\nexc expected_syncount = %f (++ %f)", *exp_cnt_exc, roundf(n_exc_targets * prob_exc + .5));
		}

	}
}

__device__ void prob_improvement(
	int* mode,
	float* alpha,
	float* beta,
	float* gamma,
	const float expected_count,
	const float error,
	const float fS,
	const float fD,
	const float alpha_delta,
	const float beta_delta,
	const float gamma_delta
	// const int group
)
{
	if (*mode == 0)
	{
		// if (group == 0) printf("\n(%d) beta=%f", group, *beta);
		// if (group == 0) printf("\n(%d) beta_delta=%f", group, beta_delta);
		*beta = fminf(*beta * fmaxf(fminf(fS / (expected_count), 1.f + beta_delta), 1.f - beta_delta), fD * (1- *gamma));
		*mode = 1;
		// if (group == 0) printf("\n(%d) beta=%f", group, *beta);
	}
	else if (*mode == 1)
	{
		// if (group == 0) printf("\n(%d) alpha=%f", group, *alpha);
		*alpha = fmaxf(*alpha + fmaxf(fminf( ( expected_count - fS) / fS, alpha_delta), -alpha_delta),
			0.05f);
		*mode = 0;
		// if (group == 0) printf("\n(%d) alpha=%f", group, *alpha);
	}

	if (error > (fS * 0.1f))
	{
		// if (group == 0) printf("\n(%d) gamma=%f", group, *gamma);
		*gamma = fminf(*gamma * fmaxf(fminf(fS / (expected_count), 1.f + gamma_delta), 1.f - gamma_delta), .3f);
		// if (group == 0) printf("\n(%d) gamma=%f", group, *gamma);
	}
	
}


__device__ int roughly_optimize_connection_probabilites(
	const float fS,
	const float fD,
	const int D,
	const int G,
	const int* G_neuron_counts, 
	const int ioffs_inh, const int ioffs_exc,
	float* p_alpha_inh, float* p_beta_inh, float* p_gamma_inh,
	float* p_alpha_exc, float* p_beta_exc, float* p_gamma_exc,
	const float alpha_delta, const float beta_delta, const float gamma_delta,
	const int group,
	const bool verbose = 1,
	const int print_group = 1
){
	
	int j = 0;

	float exp_cnt_inh = 0.f;
	float exp_cnt_exc = 0.f;
		
	int mode_inh = 0;
	int mode_exc = 0;

	float error_inh = fS;
	float error_exc = fS;
	const float p = (1. / fS);


	while (((error_inh > p) || (error_exc > p)) && (j < 300))
	{
		expected_syn_count(
			fD, 
			D, 
			G, 
			G_neuron_counts,
			ioffs_inh, ioffs_exc,
			*p_alpha_inh, *p_beta_inh, *p_gamma_inh,
			*p_alpha_exc, *p_beta_exc, *p_gamma_exc,
			&exp_cnt_inh, &exp_cnt_exc,
			group,
			verbose, print_group
		);

		error_inh = fabsf(exp_cnt_inh - fS);
		error_exc = fabsf(exp_cnt_exc - fS);
		
		j++;
		
		if ((error_inh > p))
		{
			prob_improvement(&mode_inh,
				p_alpha_inh, p_beta_inh, p_gamma_inh,
			 	exp_cnt_inh, error_inh,
			 	fS, fD,
			 	alpha_delta, beta_delta, gamma_delta
				//, group 
			);
		}
		if ((error_exc > p))
		{
			prob_improvement(&mode_exc,
				p_alpha_exc, p_beta_exc, p_gamma_exc,
				exp_cnt_exc, error_exc,
				fS, fD,
				alpha_delta, beta_delta, gamma_delta
				//, group
			);
		}

		if ((verbose) && (group == print_group))
		{
			printf("\n\n0 (%d, %d) expected_count_inh %f, expected_count_exc %f, modes %d, %d",
				group, j, exp_cnt_inh, exp_cnt_exc, 
				mode_inh, mode_exc);
			// if ((error_inh > p))
				printf("\n1 (%d, %d) alpha_inh %f, beta_inh %f , gamma_inh %f  \nerror=%f",
					group, j, *p_alpha_inh, *p_beta_inh, *p_gamma_inh,  exp_cnt_inh - fS);
			// if ((error_exc > p))
				printf("\n2 (%d, %d) alpha_exc %f, beta_exc %f , gamma_exc %f  \nerror=%f",
					group, j, *p_alpha_exc, *p_beta_exc, *p_gamma_exc,  exp_cnt_exc - fS);
		}

	}

	return j;
}

__global__ void fill_G_exp_ccsyn_per_src_type_and_delay_(
	const int S,
	const float fS,
	const int D,
	const float fD,
	const int G,
	const int* G_neuron_counts,
	float* G_conn_probs,
	int* G_exp_ccsyn_per_src_type_and_delay,
	bool verbose = 0,
	int print_group = 1
)
{
	// connection probabilities from inh -> exc
	// connection probabilities from exc -> (inh & exc)

	const int g = threadIdx.x + blockDim.x * blockIdx.x;

	if (g < G)
	{
		const int ioffs_inh = 2 * G + g;
		const int ioffs_exc = (2 + D) * G + g;

		float alpha_inh = 2.f;
		float alpha_exc = 1.f;
		float beta_inh = 1.f + fD / 3.f;
		float beta_exc = 1.f;
		float gamma_inh = .01f;
		float gamma_exc = .125f;

		const float alpha_delta = 0.04f;
		const float beta_delta = 0.04f;
		const float gamma_delta = 0.01f;



		const int opt_runs = roughly_optimize_connection_probabilites(
				fS,
				fD, 
				D,
				G,
				G_neuron_counts, 
				ioffs_inh, ioffs_exc,
				&alpha_inh, &beta_inh, &gamma_inh,
				&alpha_exc, &beta_exc, &gamma_exc,
				alpha_delta, beta_delta, gamma_delta, 
				g, 
				verbose);

		if ((g < 10) && (opt_runs > 98) || ((g == print_group) && (verbose))) {
			printf("\n(GPU: optimize_connection_probabilites) group(%d, opt_runs) = %d", g, opt_runs);
		}
		// if ((verbose) && (g == print_group)) {
		// 	printf("\nalpha_inh = %f, beta_inh = %f, gamma_inh = %f", alpha_inh, beta_inh, gamma_inh);
		// 	printf("\nalpha_exc = %f, beta_exc = %f, gamma_exc = %f", alpha_exc, beta_exc, gamma_exc);
		// }

		int expected_synapses_inh = 0;
		int expected_synapses_exc = 0;

		// int delay_with_min_exp_inh_syn_ge1 = 0
		int delay_with_max_inh_targets = 0;
		int exp_inh_syn_with_max_targets = 0;
		int max_inh_targets = 0;
		// int delay_with_min_exp_exc_syn_ge1 = 0
		int delay_with_max_exc_targets = 0;
		int exp_exc_syn_with_max_targets = 0;
		int max_exc_targets = 0;

		int idx = 0;
		int exc_syn_count = 0;
		int inh_syn_count = 0;

		for (int delay = 0; delay < D; delay++)
		{
			const float fdelay = __int2float_rn(delay);
			float prob_inh = sigmoidal_connection_probability(fdelay, fD, alpha_inh, beta_inh, gamma_inh);
			float prob_exc = sigmoidal_connection_probability(fdelay, fD, alpha_exc, beta_exc, gamma_exc);

			G_conn_probs[(g)*D + delay] = prob_inh;
			G_conn_probs[(G * D) + (g * D) + delay] = prob_exc;

			idx = delay * G;
			const int n_inh_targets = G_neuron_counts[idx + ioffs_exc];
			int n_exc_targets = n_inh_targets + G_neuron_counts[idx + ioffs_inh];
			const float f_n_inh_targets = __int2float_rn(n_inh_targets);
			float f_n_exc_targets = __int2float_rn(n_exc_targets);

			if ((delay == 0) && (G_neuron_counts[ioffs_inh - G] > 0))
			{
				// only exc neurons will have a technical probability > 0 to form an autapse
				n_exc_targets-=1;
				f_n_exc_targets-= 1.f;
			}

			inh_syn_count = min(__float2int_ru(prob_inh * f_n_inh_targets), n_inh_targets);
			expected_synapses_inh += inh_syn_count;
			G_exp_ccsyn_per_src_type_and_delay[g + idx + G] = expected_synapses_inh;
			
			idx += (D + 1) * G;
			exc_syn_count = __float2int_ru(prob_exc * f_n_exc_targets);
			expected_synapses_exc += exc_syn_count;
			G_exp_ccsyn_per_src_type_and_delay[g + idx + G] = expected_synapses_exc;

			if ((n_inh_targets > max_inh_targets)){
				exp_inh_syn_with_max_targets = inh_syn_count;
				delay_with_max_inh_targets = delay;
				max_inh_targets = n_inh_targets;
			}
			if ((n_exc_targets > max_exc_targets)){
				exp_exc_syn_with_max_targets = exc_syn_count;
				delay_with_max_exc_targets = delay;
				max_exc_targets = n_exc_targets;
			}

			// expected_synapses_inh += min(__float2int_ru(prob_inh * f_n_inh_targets), n_inh_targets);
			if ((verbose) && (g == print_group)) {
				printf("\nexp inh %f", prob_inh * f_n_inh_targets);
				printf("\nexp exc %f -> %d | %f (sum=%d)", 
					prob_exc * f_n_exc_targets, 
					min(__float2int_ru(prob_exc * f_n_exc_targets), n_exc_targets),
					roundf(prob_exc * f_n_exc_targets + .5),
					expected_synapses_exc
				);  
			}
		}
		
		// int res_inh = G_exp_ccsyn_per_src_type_and_delay[g + idx - (D * G)];
		// int res_exc = G_exp_ccsyn_per_src_type_and_delay[g + idx + G];

		if ((expected_synapses_inh != S)){
			int add = S - expected_synapses_inh;
			if (expected_synapses_inh > S){
				if (exp_inh_syn_with_max_targets < 1)  
				{
					add = 0;
					printf("\n(GPU: optimize_connection_probabilites) delay_inh(g=%d, exp_too_low=%d, max_targets=%d) = %d", 
					       g, exp_inh_syn_with_max_targets, max_inh_targets, delay_with_max_inh_targets);
				}
			} else if (exp_inh_syn_with_max_targets >= max_inh_targets){
				add = 0;
				printf("\n(GPU: optimize_connection_probabilites) delay_inh(g=%d, exp_too_high=%d, max_targets=%d) = %d", 
					   g, exp_inh_syn_with_max_targets, max_inh_targets, delay_with_max_inh_targets);
			}
			if (add != 0){
				for (int delay = delay_with_max_inh_targets; delay < D; delay++){
					G_exp_ccsyn_per_src_type_and_delay[g + (delay + 1) * G] += add;
				}
			}
			// printf("\n(%d) %d -> %d ", g, expected_synapses_inh, G_exp_ccsyn_per_src_type_and_delay[g + idx + G]);
		}

		if (expected_synapses_exc != S){
			int add = S - expected_synapses_exc;
			if (expected_synapses_exc > S){
				if (exp_exc_syn_with_max_targets < 1)  
				{
					add = 0;
					printf("\n(GPU: optimize_connection_probabilites) delay_exc(g=%d, exp_too_low=%d, max_targets=%d) = %d", 
						   g, exp_exc_syn_with_max_targets, max_exc_targets, delay_with_max_exc_targets);
				}
			} else if (exp_exc_syn_with_max_targets >= max_exc_targets){
				add = 0;
				printf("\n(GPU: optimize_connection_probabilites) delay_exc(g=%d, exp_too_high=%d, max_targets=%d) = %d", 
					   g, exp_exc_syn_with_max_targets, max_exc_targets, delay_with_max_exc_targets);
			} 
			if (add != 0){
				for (int delay = delay_with_max_exc_targets; delay < D; delay++){
					G_exp_ccsyn_per_src_type_and_delay[ g + (delay + 2 + D) * G] += add;
				}
			}
			if (G_exp_ccsyn_per_src_type_and_delay[g + (2 * D + 1) * G] != S){
				printf("\n(GPU: optimize_connection_probabilites) add(g=%d, exp=%d, max_targets=%d) = %d (%d, %d)", 
					   g, exp_exc_syn_with_max_targets, max_exc_targets, add, expected_synapses_exc,
					   G_exp_ccsyn_per_src_type_and_delay[g + (2 * D + 1) * G]);
			}

		} 

		if ((verbose) && (g == print_group)) {
			printf("\nres_inh = %d", expected_synapses_inh);
			printf("\nres_exc = %d", expected_synapses_exc);
		}
	}
}



void fill_G_exp_ccsyn_per_src_type_and_delay(
	const int S,
	const int D,
	const int G,
	const int* G_neuron_counts,
	float* G_conn_probs,
	int* G_exp_ccsyn_per_src_type_and_delay,
	bool verbose
)
{	
	cudaDeviceSynchronize();
	LaunchParameters launch(G, (void *)fill_G_exp_ccsyn_per_src_type_and_delay_); 

	fill_G_exp_ccsyn_per_src_type_and_delay_ KERNEL_ARGS2(launch.grid3, launch.block3)(
		S,
		static_cast<float>(S),
		D,
		static_cast<float>(D),
		G,
		G_neuron_counts,
		G_conn_probs,
		G_exp_ccsyn_per_src_type_and_delay,
		verbose
	);
	
	cudaDeviceSynchronize();
	printf("\n\n");
}


__device__ int relative_typed_delay_rep_index(
	const int N_autapse_idx,
	const int G_rep_idx0,
	const int G_rep_idx1,
	const int g_N_count,
	const int* G_rep,
	const int n_groups,
	const int* cc_snk,
	bool verbose

)
{
	if (g_N_count == 0)
	{
		return -1;
	}

	int G_rep_idx = G_rep_idx0;
	int g = G_rep[G_rep_idx];
	int Ng_start = cc_snk[g];
	const int Ng_last = cc_snk[G_rep[G_rep_idx1] +1];

	
	if ((N_autapse_idx < Ng_start) || (N_autapse_idx >= Ng_last))
	{
		return -1;
		if (verbose)
		{
			printf(
			"(search, not in range) g=(%d), n=%d, G_rep[%d: %d], Ng_start=%d, Ng_last=%d\n", 
			g, N_autapse_idx, G_rep_idx0, G_rep_idx1, Ng_start, Ng_last);
		}
	}

	int result = N_autapse_idx;
	result -= Ng_start;
	int Ng_next = cc_snk[g + 1];

	if (verbose)
	{
		printf("(search) g=(%d), n=%d, Ng_start=%d, Ng_next=%d\n", g, N_autapse_idx, Ng_start, Ng_next);
	}
	//if (bprint)
	//{
	//	printf("\n  search (%d) g_start_col %d, n_g_search %d, g=%d",
	//		N_autapse_idx, g_search_start_col, n_g_search, g );
	//	printf("\n  (%d)  [%d], src_loc %d, g = %d [%d, %d]... %d]",
	//		N_autapse_idx, result, src_loc, g, start_col_next_group, end_col_next_group, last_col);
	//}
	
	bool found = (N_autapse_idx >= Ng_start) && (N_autapse_idx < Ng_next);

	int Ng_prev = Ng_next;
	
	while ((!found) && (G_rep_idx < G_rep_idx1))
	{
		G_rep_idx++;
		
		g = G_rep[G_rep_idx];
		Ng_start = cc_snk[g];
		Ng_next = cc_snk[g + 1];

		result -= (Ng_start - Ng_prev);

		found = (N_autapse_idx >= Ng_start) && (N_autapse_idx < Ng_next);
		if (verbose)
		{
			printf("(search, found=%d) g=(%d), n=%d, Ng_start=%d, Ng_next=%d\n", found, g, N_autapse_idx, Ng_start, Ng_next);
		}
	}
	return result * found + (-1) * (!found);
}


__global__ void fill_relative_autapse_indices_(
	const int D,
	const int G,
	const int* cc_src,
	const int* cc_snk,
	const int* G_rep,
	const int* G_delay_counts,
	int* G_autapse_indices,
	int* G_relative_autapse_indices,
	bool verbose = 0,
	int print_group = 1
)
{
	const int g = blockIdx.x * blockDim.x + threadIdx.x;  // NOLINT(bugprone-narrowing-conversions, cppcoreguidelines-narrowing-conversions)
	
	if (g < G)
	{
		
		const int N_autapse_idx = cc_src[g];
		const int g_N_count = cc_src[g + 1] - N_autapse_idx;
	
		for (int d=0; d < D; d++)
		{
			const int g_rep_col0 = G_delay_counts[g * (D + 1) + d];
			const int g_rep_col1 = G_delay_counts[g * (D + 1) + d+1]-1;
			const int G_rep_idx0 = g * G + g_rep_col0;
			const int G_rep_idx1 = g * G + g_rep_col1;
			const int n_groups = g_rep_col1 - g_rep_col0 +1;

			if (verbose && (g == print_group))
			{
				printf("g=(%d), n=%d, d=%d, g_rep_cols=[%d, %d], idcs=[%d,%d], groups=[%d, ...,%d]\n", 
					g, N_autapse_idx, d, 
					g_rep_col0, g_rep_col1,
					G_rep_idx0, G_rep_idx1,
					G_rep[G_rep_idx0], G_rep[G_rep_idx1]);
			}

			const int relative_autapse_index = relative_typed_delay_rep_index(
				N_autapse_idx,
				G_rep_idx0,
				G_rep_idx1,
				g_N_count,
				G_rep,
				n_groups,
				cc_snk,
				verbose && (g == print_group)
			);

			G_relative_autapse_indices[g + d * G] = relative_autapse_index;

			if (relative_autapse_index != -1){
				G_autapse_indices[g + d * G] = N_autapse_idx;
			} else {
				G_autapse_indices[g + d * G] = -1;
			}

			if (verbose && (g == print_group))
			{
				printf("g=(%d), n=%d, d=%d, N=%d, rN=%d\n", 
					g, N_autapse_idx, d, 
					G_autapse_indices[g + d * G],
					relative_autapse_index);
			}

			//if (bprint)
			//{
			//	printf("\nres: %d << %d, g = %d, d = %d\n---------\n", relative_self_index[g + d * G], self_index[g + d * G],g,d);
			//}

		}
	}
}

__forceinline__ __device__ int random_uniform_int(curandState *local_state, const float min, const float max)
{
	return __float2int_rd(fminf(min + curand_uniform(local_state) * (max-min +1.f), max));
}

__global__ void k_set_locally_indexed_connections(
	const int N,
	const int S,
	const int D,
	const int G,
	curandState* state,
	const int* N_G,
	const int* cc_src,
	int* N_delays,
	const int* cc_syn,
	const int* G_neuron_counts,
	const int* G_relative_autapse_indices,
	const int gc_location0,
	const int gc_location1,
	const int gc_conn_shape0,
	const int gc_conn_shape1,
	bool b_autapses,
	const float init_weight,
	float* weights,
	int* sort_keys,
	int* network_rep
)
{
	extern __shared__ int sh_delays[];
	int* max_delay_counts = &sh_delays[(D+1) * blockDim.x];

	const int n = starting_row + blockIdx.x * blockDim.x + threadIdx.x;
	
	if (n < starting_row + n_rows)
	{
		curandState local_state = state[n];
		
		const int src_loc = neuron_groups[N + n];
		int syn_count_idx = threadIdx.x;
		const int row_start_idx = n * S;

		sh_delays[syn_count_idx] = 0;
		delays[n] = 0;

		for (int d=1; d<D+1; d++)
		{
			//int start_rep_col = synapse_count_per_delay[src_loc + (d-1) * G];
			int end_rep_col = synapse_count_per_delay[src_loc + d * G];
			
			sh_delays[syn_count_idx + d* blockDim.x] = end_rep_col + starting_write_col;

			max_delay_counts[syn_count_idx + (d-1)* blockDim.x] = G_neuron_counts[src_loc + (d-1) * G];

			delays[n + d * N] += end_rep_col;
		}

		int delay = 0;
		syn_count_idx = threadIdx.x;

		int sort_key = row_start_idx + starting_write_col + __float2int_rn(fmaxf(0.f, (D - S) * n));
		int delay_start_rep_col = sh_delays[syn_count_idx];
		int delay_end_rep_col = sh_delays[syn_count_idx + blockDim.x];
		


		float offset_max = 0.f;
		float offset_min = 0.f;

		int min = 0;
		int max = max_delay_counts[syn_count_idx] - 1;
		float fmax_idx = __int2float_rn(max);
			
		float frange = 0.f;
		int n_rep_cols = (delay_end_rep_col)-delay_start_rep_col;
		if (n_rep_cols > 1)
		{
			frange = (fmax_idx+1.f) / __int2float_rn(n_rep_cols);
		}
		

		//float fmin;
		float fmax;

		const float fmin_range = 2.f; // guaranties to find an alternative to autapse

		
		if ((frange> fmin_range ))
		{
			offset_min = 0.f;
			fmax = frange -1.f;
			offset_max = fmax_idx - roundf(fmax);
		}
		
		//int n_range_cols = max - min;
		
		int self_sink;
		if (!allow_self_connection)
			self_sink = local_autapse_indices[src_loc + delay * G] + (n - typed_cumulative_count_src[src_loc]);
		
		int i = 0;

		//bool bprint = (n==25) || (n==0);
		//bool bprint = false;
		
		for (int write_col = starting_write_col; write_col < n_write_cols+starting_write_col; write_col++)
		{
			const int write_idx = row_start_idx + write_col;

			while ((write_col == delay_end_rep_col) && (delay < D+1))
			{
				
				syn_count_idx += blockDim.x;
				delay_start_rep_col = sh_delays[syn_count_idx];
				delay_end_rep_col = sh_delays[syn_count_idx + blockDim.x];
				delay++;

				n_rep_cols = delay_end_rep_col-delay_start_rep_col;

				if (n_rep_cols >0)
				{
					offset_max = 0.f;
					offset_min = 0.f;
					min = 0;
					max = max_delay_counts[syn_count_idx] - 1;
					fmax_idx = __int2float_rn(max);

					sort_key = write_idx;
					i = 0;
				
					//if (delay >= D + 1)
					//{
					//	printf("\n Warning (k_set_locally_indexed_connections): Loop limit exceeded!");
					//}
					
					frange = (fmax_idx + 1) / __int2float_rn(n_rep_cols);
					if (frange > fmin_range)
					{	
						offset_min = 0.f;
						fmax = frange-1.f;
						offset_max = fmax_idx - roundf(fmax);
					}
				}
				
			}

			//if (bprint)
			//{
			//	printf("\n\n start (%d, %d) [%d,%d] (d: %d) range (%f, %f) [%f] max: %f", n, write_col, delay_start_rep_col, delay_end_rep_col, delay, offset_min, fmax_idx - offset_max, frange, fmax_idx);
			//}

			int new_sink = random_uniform_int(&local_state, offset_min, fmax_idx - offset_max);
			while ((!allow_self_connection) && (delay == 0) && (new_sink==self_sink) && (i<50))
			{
				new_sink = random_uniform_int(&local_state, offset_min, fmax_idx - offset_max);
				i++;
				if (i >= 50)
				{
					printf("\n (%d, %d) Warning (k_set_locally_indexed_connections, self_sink: max_count %f): Loop limit exceeded!", n, write_col, fmax_idx);
					printf("\n (%d, %d) Warning [...,%d] (d: %d) new_sink %d, self_sink %d, idx %d", n, write_col, delay_end_rep_col, delay, new_sink, self_sink, write_idx);
				}
			}
			
			if (!(frange > fmin_range))
			{
				int k = 0;
				
				bool duplicated = true;
				while (duplicated && (k<50))
				{
					if ((k==45))
					{
						printf("\n (%d, %d) Warning (duplicated): Loop limit exceeded!", n, write_col);
						printf("\n (%d, %d) [%d,%d] (d: %d) new_sink %d range (%f, %f) [%f]  max: %f",
							       n, write_col, delay_start_rep_col, delay_end_rep_col, delay, new_sink, offset_min, fmax_idx - offset_max, frange, fmax_idx);
					}
					duplicated = false;
					int j = row_start_idx + delay_start_rep_col;
					i = delay_start_rep_col;
					while ((!duplicated) && (i < write_col))
					{
						duplicated = (network_rep[j] == new_sink);
						if ((k >= 45))
						{
							printf("\nn = (%d, %d) (%d, %d, %d) %d <%d> %d", n, write_col, i, j, k, network_rep[j], duplicated , new_sink);
						}
						i++;
						j++;
					}
					if (duplicated)
					{
						new_sink = random_uniform_int(&local_state, offset_min, fmax_idx - offset_max);
						i = 0;
						while ((!allow_self_connection) && (delay == 0) && (new_sink == self_sink) && (i < 50))
						{
							
							new_sink = random_uniform_int(&local_state, offset_min, fmax_idx - offset_max);
							i++;
							if (i >= 50)
							{
								printf("\n (%d, %d) Warning (2) (k_set_locally_indexed_connections, self_sink: max_count %f): Loop limit exceeded!", n, write_col, fmax_idx);
								printf("\n (%d, %d) Warning (2) [...,%d] (d: %d) new_sink %d, self_sink %d", n, write_col, delay_end_rep_col, delay, new_sink, self_sink);
							}
						}
					}
					k++;
				}
			}

			//if (bprint)
			//{
			//	printf("\n end (%d, %d) [%d,%d] (d: %d) new_sink %d range (%f, %f) [%f]  max: %f ", 
			//       n, write_col, delay_start_rep_col, delay_end_rep_col, delay, new_sink, offset_min, fmax_idx - offset_max, frange, fmax_idx);
			//}

			if (new_sink > max)
			{
				printf("\n Warning (%d, %d) (k_set_locally_indexed_connections): Max index exceeded! [%d,%d] (d: %d) new_sink %d range (%f, %f) [%f]  max: %f / %d",
					n, write_col, delay_start_rep_col, delay_end_rep_col, delay, new_sink, offset_min, fmax_idx - offset_max, frange, fmax_idx, max);
			}


			if (frange > fmin_range)
			{
				offset_min = roundf(fmax) + 1.f;

				fmax += frange;

				offset_max = fmax_idx - roundf(fmax);
			}
			else
			{
				if (new_sink == max)
				{
					max--;
					offset_max += 1.f;

				}
				if (new_sink == min)
				{
					min++;
					offset_min += 1.f;
				}
			}

			//if (limit_exceeded==true && (limit_exceeded2 == false))
			//{
			//	limit_exceeded2 = true;
			//	write_col -= (delay_end_rep_col - delay_start_rep_col);
			//	bprint = true;
			//}



			//delay_rep[write_idx] = delay;
			//snk_type_rep[write_idx] = snk_type;
			//network_rep5[write_idx] = self_sink * (delay == 0) - (delay != 0);
			
			sort_keys[write_idx] = sort_key;
			network_rep[write_idx] = new_sink;
			syn_info[write_idx] = init_weight;
		
		}
		
		state[n] = local_state;
	}
}

void fill_N_rep(
	const int N,
	const int S,
	const int D,
	const int G,
	curandState* curand_states,
	const int n_curand_states,
	const int* N_G,
	const int* cc_src,
	const int* cc_snk,
	const int* G_rep,
	const int* G_neuron_counts,
	const int* G_group_delay_counts,
	int* G_autapse_indices,
	int* G_relative_autapse_indices,
	const int gc_location0,
	const int gc_location1,
	const int gc_conn_shape0,
	const int gc_conn_shape1,
	int* N_rep,
	bool verbose
)
{
	printf("\nConnecting: ((%d, %d), (%d, %d))", gc_location0, gc_location1, gc_conn_shape0, gc_conn_shape1);
	cudaDeviceSynchronize();
	LaunchParameters launch(G, (void *)fill_relative_autapse_indices_); 
	fill_relative_autapse_indices_ KERNEL_ARGS2(launch.grid3, launch.block3)(
		D,
		G,
		cc_src,
		cc_snk,
		G_rep,
		G_group_delay_counts,
		G_autapse_indices,
		G_relative_autapse_indices,
		false);
	cudaDeviceSynchronize();

	LaunchParameters l(gc_conn_shape0, (void*)k_set_locally_indexed_connections);
	cudaDeviceSynchronize();
	
	k_set_locally_indexed_connections KERNEL_ARGS3(l.grid3, l.block3, l.block3.x* ((1 + 3 * D)) * sizeof(int))(
		N,
		S,
		D,
		G,
		curand_states,
		N_G,
		cc_src,
		N_delays,
		cc_syn,
		G_neuron_counts,
		G_relative_autapse_indices,
		gc_location0,
		gc_location1,
		gc_conn_shape0,
		gc_conn_shape1,
		group_conn.src_type != group_conn.snk_type
		group_conn.initial_weight,
		weights,
		sort_keys,
		N_rep
	  );
}