#include <pybind11/include/pybind11/pybind11.h>
#include <pybind11/include/pybind11/numpy.h>
#include <pybind11/include/pybind11/stl.h>

#include <simulation/snn_simulation.cuh>


namespace py = pybind11;


SnnSimulation make_SnnSimulation(
    const int N,
    const int G,
    const int S,
    const int D,
    const int T,
    const int n_voltage_plots,
    const int voltage_plot_length,
    const long voltage_plot_data_dp,
    const long voltage_plot_map_dp,
    const int n_scatter_plots,
    const int scatter_plot_length,
    const long scatter_plot_data_dp,
    const long scatter_plot_map_dp,
    std::shared_ptr<CuRandStates> curand_states,
    const long N_pos_dp,
    const long N_G_dp,
    const long G_props_dp, 
    const long N_rep_dp, 
    const long N_delays_dp, 
    const long N_states_dp,
    const long N_weights_dp,
    const long fired_dp,
    const long last_fired_dp,
    const long firing_times_dp,
    const long firing_idcs_dp,
    const long firing_counts_dp,
    const long G_stdp_config0_dp,
    const long G_stdp_config1_dp
){
    float* voltage_plot_data = reinterpret_cast<float*> (voltage_plot_data_dp);
    int* voltage_plot_map = reinterpret_cast<int*> (voltage_plot_map_dp);
    float* scatter_plot_data = reinterpret_cast<float*> (scatter_plot_data_dp);    
    int* scatter_plot_map = reinterpret_cast<int*> (scatter_plot_map_dp);    

    float* N_pos = reinterpret_cast<float*> (N_pos_dp);
    int* N_G = reinterpret_cast<int*> (N_G_dp);
    float* G_props = reinterpret_cast<float*> (G_props_dp);
    int* N_rep = reinterpret_cast<int*> (N_rep_dp);

    // int* N_rep_pre_synaptic = reinterpret_cast<int*> (N_rep_pre_synaptic_dp);
    // int* N_rep_pre_synaptic_counts = reinterpret_cast<int*> (N_rep_pre_synaptic_counts_dp);
    // int* N_rep_pre_synaptic_counts = reinterpret_cast<int*> (N_rep_dp);
    
    int* N_delays = reinterpret_cast<int*> (N_delays_dp);
    float* N_states = reinterpret_cast<float*> (N_states_dp);
    float* N_weights = reinterpret_cast<float*> (N_weights_dp);

    float* fired = reinterpret_cast<float*> (fired_dp);
    int* last_fired = reinterpret_cast<int*> (last_fired_dp);
    float* firing_times = reinterpret_cast<float*> (firing_times_dp);
    int* firing_idcs = reinterpret_cast<int*> (firing_idcs_dp);
    int* firing_counts = reinterpret_cast<int*> (firing_counts_dp);

    int* G_stdp_config0 = reinterpret_cast<int*> (G_stdp_config0_dp);
    int* G_stdp_config1 = reinterpret_cast<int*> (G_stdp_config0_dp);
    
    return SnnSimulation(
        N,
        G,
        S,
        D,
        T,
        n_voltage_plots,
        voltage_plot_length,
        voltage_plot_data,
        voltage_plot_map,
        n_scatter_plots,
        scatter_plot_length,
        scatter_plot_data,
        scatter_plot_map,
        curand_states->states,
        N_pos,
        N_G,
        G_props, 
        N_rep, 
        // N_rep_pre_synaptic, 
        // N_rep_pre_synaptic_counts, 
        N_delays, 
        N_states,
        N_weights,
        fired,
        last_fired,
        firing_times,
        firing_idcs,
        firing_counts,
        G_stdp_config0,
        G_stdp_config1
    );
}



PYBIND11_MODULE(snn_simulation_gpu, m)
    {
        
    m.def("print_random_numbers", &print_random_numbers2);
    
    py::class_<SnnSimulation, std::shared_ptr<SnnSimulation>>(m, "SnnSimulation_") //, py::dynamic_attr())
    //.def(py::init<int>())
    .def_readonly("N", &SnnSimulation::N)
    .def_readonly("G", &SnnSimulation::G)
    .def_readonly("S", &SnnSimulation::S)
    .def_readonly("D", &SnnSimulation::D)
    .def_readonly("t", &SnnSimulation::t)
    .def_readonly("N_G", &SnnSimulation::N_G)
    .def_readwrite("stdp_active", &SnnSimulation::stdp_active)
    .def("update", &SnnSimulation::update)
    .def("swap_groups", &SnnSimulation::swap_groups_python)
    .def("set_pre_synaptic_pointers", &SnnSimulation::set_pre_synaptic_pointers_python,
        py::arg("N_rep_pre_synaptic"),
        py::arg("N_rep_pre_synaptic_weight_idx"),
        py::arg("N_rep_pre_synaptic_counts"))
    .def("set_stdp_config", &SnnSimulation::set_stdp_config, 
        py::arg("stdp_config_id"), 
        py::arg("activate") = true)
    .def("actualize_N_rep_pre_synaptic", &SnnSimulation::actualize_N_rep_pre_synaptic)
    .def("__repr__",
        [](const SnnSimulation &sim) {
            return "SnnSimulation(N=" + std::to_string(sim.N) + ")";
        })
        ;
    m.def("SnnSimulation", &make_SnnSimulation,
        py::arg("N"),
        py::arg("G"),
        py::arg("S"),
        py::arg("D"),
        py::arg("T"),
        py::arg("n_voltage_plots"),
        py::arg("voltage_plot_length"),
        py::arg("voltage_plot_data"),
        py::arg("voltage_plot_map"),
        py::arg("n_scatter_plots"),
        py::arg("scatter_plot_length"),
        py::arg("scatter_plot_data"),
        py::arg("scatter_plot_map"),
        py::arg("curand_states_p"),
        py::arg("N_pos"),
        py::arg("N_G"),
        py::arg("G_props"),
        py::arg("N_rep"),
        // py::arg("N_rep_pre_synaptic"),
        // py::arg("N_rep_pre_synaptic_c"),
        py::arg("N_delays"),
        py::arg("N_states"),
        py::arg("N_weights"),
        py::arg("fired"),
        py::arg("last_fired"),
        py::arg("firing_times"),
        py::arg("firing_idcs"),
        py::arg("firing_counts"),
        py::arg("G_stdp_config0"),
        py::arg("G_stdp_config1")
    );
}