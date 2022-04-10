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
    float* G_props; 
    int* N_rep; 
    int* N_delays; 
    float* N_states; 
    
    float* N_weights;
    
    float* fired; 
    
    float* firing_times_write;
    float* firing_times_read;
    float* firing_times;

    int* firing_idcs_write;
    int* firing_idcs_read;
    int* firing_idcs;
    
    int* firing_counts_write;
    int* firing_counts;
    
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
        float* G_props_, 
        int* N_rep_, 
        int* N_delays_, 
        float* N_states_,
        float* N_weights_,
        float* fired_,
        float* firing_times_,
        int* firing_idcs_,
        int* firing_counts_
    );
    
    void update_voltage_plot();
    void print_info(bool bprint_idcs = false);
    void update(bool verbose);
};