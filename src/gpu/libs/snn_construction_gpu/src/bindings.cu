#include <pybind11/include/pybind11/pybind11.h>
#include <pybind11/include/pybind11/numpy.h>
#include <pybind11/include/pybind11/stl.h>

#include <example/snn_engine_gpu.cuh>
#include <utils/launch_parameters.cuh>
#include <snn_construction.cuh>


namespace py = pybind11;


// void print_dict(const py::dict& dict) {
//     /* Easily interact with Python types */
//     for (auto item : dict)
//         std::cout << "key=" << std::string(py::str(item.first)) << ", "
//                   << "value=" << std::string(py::str(item.second)) << std::endl;
// }


void fill_N_G_group_id_and_G_neuron_count_per_type_python(
    const int N, 
    const int G, 
    const long N_pos_dp,
    const py::tuple& N_pos_shape,
    long N_G_dp,
    const py::tuple& G_shape,
    long G_neuron_counts_dp,
    const int N_pos_n_cols,
    const int N_G_n_cols,
	const int N_G_neuron_type_col,
	const int N_G_group_id_col
){
    const float* N_pos = reinterpret_cast<float*> (N_pos_dp);
	int* N_G = reinterpret_cast<int*> (N_G_dp);
	int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);

    fill_N_G_group_id_and_G_neuron_count_per_type(
        N, 
        G, 
        N_pos, 
        N_pos_shape[0].cast<int>(), N_pos_shape[1].cast<int>(), N_pos_shape[2].cast<int>(),
        N_G, 
        G_neuron_counts,
        G_shape[0].cast<int>(), G_shape[1].cast<int>(), G_shape[2].cast<int>(),
        N_pos_n_cols,
        N_G_n_cols,
        N_G_neuron_type_col,
        N_G_group_id_col
    );
}


void fill_G_neuron_count_per_delay_python(
	const int S,
	const int D,
	const int G,
	const long G_delay_distance_dp,
	long G_neuron_counts_dp
){
    
    const int* G_delay_distance = reinterpret_cast<int*> (G_delay_distance_dp);
	int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);

    fill_G_neuron_count_per_delay(
        S,
        D,
        G,
        G_delay_distance,
        G_neuron_counts
    );
}



void fill_G_exp_ccsyn_per_src_type_and_delay_python(
	const int S,
	const int D,
	const int G,
    const long G_neuron_counts_dp,
    long G_conn_probs_dp,
	long G_exp_ccsyn_per_src_type_and_delay_dp
){
    
	const int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);
	float* G_conn_probs = reinterpret_cast<float*> (G_conn_probs_dp);
	int* G_exp_ccsyn_per_src_type_and_delay = reinterpret_cast<int*> (G_exp_ccsyn_per_src_type_and_delay_dp);

    fill_G_exp_ccsyn_per_src_type_and_delay(
        S,
        D,
        G,
        G_neuron_counts,
        G_conn_probs,
        G_exp_ccsyn_per_src_type_and_delay
    );
}

void fill_N_rep_python(
	const int N,
	const int S,
	const int D,
	const int G,
	const long N_G_dp,
	const long cc_src_dp,
	const long cc_snk_dp,
	const long G_rep_dp,
	const long G_neuron_counts_dp,
	const long G_delay_counts_dp,
	long autapse_indices_dp,
	long relative_autapse_indices_dp,
	long N_rep_dp,
	bool verbose = 0
)
{
    const int* N_G = reinterpret_cast<int*> (N_G_dp);
    const int* cc_src = reinterpret_cast<int*> (cc_src_dp);
    const int* cc_snk = reinterpret_cast<int*> (cc_snk_dp);
    const int* G_rep = reinterpret_cast<int*> (G_rep_dp);
    const int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);
    const int* G_delay_counts = reinterpret_cast<int*> (G_delay_counts_dp);
    int* autapse_indices = reinterpret_cast<int*> (autapse_indices_dp);
    int* relative_autapse_indices = reinterpret_cast<int*> (relative_autapse_indices_dp);
    int* N_rep = reinterpret_cast<int*> (N_rep_dp);

    fill_N_rep(
        N, 
        S, 
        D, 
        G,
        N_G,
        cc_src, cc_snk,
        G_rep, G_neuron_counts, G_delay_counts,
        autapse_indices, 
        relative_autapse_indices,
        N_rep,
        verbose
    );
}




PYBIND11_MODULE(snn_construction_gpu, m)
{
    m.def("fill_N_G_group_id_and_G_neuron_count_per_type", 
          &fill_N_G_group_id_and_G_neuron_count_per_type_python, 
          py::arg("N"),
          py::arg("G"),
          py::arg("N_pos"),
          py::arg("N_pos_shape"),
          py::arg("N_G"),
          py::arg("G_shape"),
          py::arg("G_neuron_counts"),
          py::arg("N_pos_n_cols") = 13, 
          py::arg("N_G_n_cols") = 3,
          py::arg("N_G_neuron_type_col") = 0,
          py::arg("N_G_group_id_col") = 1
    );
    
    m.def("fill_G_neuron_count_per_delay", 
          &fill_G_neuron_count_per_delay_python, 
          py::arg("S"),
          py::arg("D"),
          py::arg("G"),
          py::arg("G_delay_distance"),
          py::arg("G_neuron_counts")
    );

    m.def("fill_G_exp_ccsyn_per_src_type_and_delay", 
          &fill_G_exp_ccsyn_per_src_type_and_delay_python, 
          py::arg("S"),
          py::arg("D"),
          py::arg("G"),
          py::arg("G_neuron_counts"),
          py::arg("G_conn_probs"),
          py::arg("G_exp_ccsyn_per_src_type_and_delay")
    );

    m.def("fill_N_rep", 
          &fill_N_rep_python, 
          py::arg("N"),
          py::arg("S"),
          py::arg("D"),
          py::arg("G"),
          py::arg("N_G_dp"),
          py::arg("cc_src_dp"),
          py::arg("cc_snk_dp"),
          py::arg("G_rep_dp"),
          py::arg("G_neuron_counts_dp"),
          py::arg("G_delay_counts_dp"),
          py::arg("autapse_indices_dp"),
          py::arg("relative_autapse_indices_dp"),
          py::arg("N_rep_dp"),
          py::arg("verbose")
);

    // m.def("pyadd", &pyadd, "A function which adds two numbers");
    // m.def("pyadd_occupancy", &pyadd_occupancy);
    
    // py::class_<CudaGLResource>(m, "CudaGLResource", py::dynamic_attr())
    //     .def(py::init())
    //     .def("map", &CudaGLResource::map)
    //     // .def("unmap", &CudaGLResource::unmap)
    //     // .def("register", &CudaGLResource::register_)
    //     // .def("unregister", &CudaGLResource::unregister)
    //     .def_readonly("id", &CudaGLResource::id)
    //     .def_readonly("size", &CudaGLResource::size)
    //     .def_readonly("is_mapped", &CudaGLResource::is_mapped)
    //     .def("__repr__",
    //         [](const CudaGLResource &a) {
    //             return "<CudaGLResource(id=" + std::to_string(a.id) + ",is_mapped=" + std::to_string(a.is_mapped) + ")>";
    //         }
    // );
}
