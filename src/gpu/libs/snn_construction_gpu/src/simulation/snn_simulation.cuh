#include <utils/curand_states.cuh>
#include <utils/launch_parameters.cuh>


struct SnnSimulation
{
    int N;
    int G;
    int S;
    int D;
    int T;

    int n_voltage_plots;
    int voltage_plot_length;
    float* voltage_plot_data;
    int* voltage_plot_map;
    bool b_update_voltage_plot = true;
    
    int n_scatter_plots;
    int scatter_plot_length;
    float* scatter_plot_data;
    int* scatter_plot_map;

    curandState* rand_states;
    
    float* N_pos; 
    int* N_G; 
    int* G_group_delay_counts; 
    float* G_props; 
    int* N_rep; 
    int* N_rep_buffer; 
    int* N_rep_pre_synaptic_idx; 
    int* N_rep_pre_synaptic_counts; 
    int* N_delays; 
    float* N_states; 
    
    float* N_weights;
    
    float* fired; 
    int* last_fired; 
    
    float* firing_times_write;
    float* firing_times_read;
    float* firing_times;

    int* firing_idcs_write;
    int* firing_idcs_read;
    int* firing_idcs;
    
    int* firing_counts_write;
    int* firing_counts;

    bool stdp_active = false;
    int* G_stdp_config0;
    int* G_stdp_config1;
    int* G_stdp_config_current;

    float* G_avg_weight_inh;
    float* G_avg_weight_exc;
    int* G_syn_count_inh;
    int* G_syn_count_exc;
    
    LaunchParameters lp_update_state;
    LaunchParameters lp_update_voltage_plot;
    LaunchParameters lp_update_scatter_plot;

    cusparseHandle_t fired_handle;
	
	cusparseSpMatDescr_t firing_times_sparse;
	cusparseDnMatDescr_t firing_times_dense;

	void* fired_buffer{nullptr};
	
	int n_fired = 0;
	size_t fired_buffer_size = 0;
	int n_fired_total = 0;
	int n_fired_total_m1 = 0;
	int n_fired_0 = 0;
	int n_fired_m1 = 0;
		
	int firing_counts_idx = 1;
	int firing_counts_idx_m1 = 1;
	// int firing_counts_idx_end = 1;

	int reset_firing_times_ptr_threshold;
	int n_fired_m1_to_end = 0;

    int t = 0;

    bool resetting = false;

    std::chrono::steady_clock::time_point t0;
    // std::chrono::steady_clock::time_point t1;
    uint update_duration;

    SnnSimulation(
        int N_,
        int G_,
        int S_,
        int D_,
        int T,

        int n_voltage_plots_,
        int voltage_plot_length_,
        float* voltage_plot_data_,
        int* voltage_plot_map_,

        int n_scatter_plots_,
        int scatter_plot_length_,
        float* scatter_plot_data_,
        int* scatter_plot_map_,
        
        curandState* rand_states_,
        float* N_pos_,
        int* N_G_,
        int* G_group_delay_counts_,
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
        
        int* G_stdp_config0_,
        int* G_stdp_config1_,
        float* G_avg_weight_inh_,
        float* G_avg_weight_exc_,
        int* G_syn_count_inh_,
        int* G_syn_count_exc_
    );
    
    void update_plots();
    void print_info(bool bprint_idcs = false);
    void update(bool verbose);

    void swap_groups(
        long* neurons, int n_neurons, 
        long* groups, int n_groups, 
        int* neuron_group_indices,
        int* G_swap_tensor, const int G_swap_tensor_shape_1,
        float* swap_rates_inh, float* swap_rates_exc,
        int* group_neuron_counts_inh, int* group_neuron_counts_exc, int* group_neuron_counts_total,
        int* G_delay_distance,
        int* N_relative_G_indices, int* G_neuron_typed_ccount,
        int* neuron_group_counts,
        int print_idx
    );
    void swap_groups_python(
        long neurons, int n_neurons, 
        long groups, int n_groups, 
        long neuron_group_indices,
        long G_swap_tensor, const int G_swap_tensor_shape_1,
        long swap_rates_inh, long swap_rates_exc,
        long group_neuron_counts_inh, long group_neuron_counts_exc, long group_neuron_counts_total, 
        long G_delay_distance,
        long N_relative_G_indices, long G_neuron_typed_ccount,
        long neuron_group_counts,
        int print_idx
    );

    void set_stdp_config(int stdp_config_id, bool activate = true);

    // void set_pre_synaptic_pointers(
    //     int* Buffer_, 
    //     int* N_rep_pre_synaptic_idx_,
    //     int* N_rep_pre_synaptic_counts_);
    // void set_pre_synaptic_pointers_python(
    //     const long Buffer_dp, 
    //     const long N_rep_pre_synaptic_idx_dp,
    //     const long N_rep_pre_synaptic_counts_dp);

    void actualize_N_rep_pre_synaptic();

    void calculate_avg_group_weight();


};