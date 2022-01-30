#include <pybind11/include/pybind11/pybind11.h>
#include <pybind11/include/pybind11/numpy.h>
#include <pybind11/include/pybind11/stl.h>

#include <example/snn_engine_gpu.cuh>
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

struct CuRandStatesPointer
{
    std::shared_ptr<CuRandStates> ptr_;
    
    CuRandStatesPointer(    
        const int n_curand_states,
        bool verbose = true
    ){
        ptr_ = std::make_shared<CuRandStates>(n_curand_states, verbose);
        //print_random_numbers2(ptr_);
    }

    int n_states(){
        return ptr_->n_states;
    }

    std::shared_ptr<CuRandStates> ptr(){
        return ptr_;
    }
};



void fill_N_rep_python(
	const int N,
	const int S,
	const int D,
	const int G,
    std::shared_ptr<CuRandStates> curand_states,
	const long N_G_dp,
	const long cc_src_dp,
	const long cc_snk_dp,
	const long G_rep_dp,
	const long G_neuron_counts_dp,
	const long G_group_delay_counts_dp,
	long G_autapse_indices_dp,
	long G_relative_autapse_indices_dp,
    bool has_autapses,
    const py::tuple& gc_location,
    const py::tuple& gc_conn_shape,
	long cc_syn_dp,
	long N_delays_dp,
    long sort_keys_dp,
	long N_rep_dp,
	bool verbose = 0
)
{
    const int* N_G = reinterpret_cast<int*> (N_G_dp);
    const int* cc_src = reinterpret_cast<int*> (cc_src_dp);
    const int* cc_snk = reinterpret_cast<int*> (cc_snk_dp);
    const int* G_rep = reinterpret_cast<int*> (G_rep_dp);
    const int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);
    const int* G_group_delay_counts = reinterpret_cast<int*> (G_group_delay_counts_dp);
    int* G_autapse_indices = reinterpret_cast<int*> (G_autapse_indices_dp);
    int* G_relative_autapse_indices = reinterpret_cast<int*> (G_relative_autapse_indices_dp);
    int* cc_syn = reinterpret_cast<int*> (cc_syn_dp);
    int* N_delays = reinterpret_cast<int*> (N_delays_dp);
    int* sort_keys = reinterpret_cast<int*> (sort_keys_dp);
    int* N_rep = reinterpret_cast<int*> (N_rep_dp);

    fill_N_rep(
        N, 
        S, 
        D, 
        G,
        curand_states->states,
        curand_states->n_states,
        N_G,
        cc_src, 
        cc_snk,
        G_rep, 
        G_neuron_counts, 
        G_group_delay_counts,
        G_autapse_indices, 
        G_relative_autapse_indices,
        has_autapses,
        gc_location[0].cast<int>(), gc_location[1].cast<int>(),
        gc_conn_shape[0].cast<int>(), gc_conn_shape[1].cast<int>(),
        cc_syn,
        N_delays,
        sort_keys,
        N_rep,
        verbose
    );
}


void sort_N_rep_python(
	const int N,
	const int S,
	long sort_keys_dp,
	long N_rep_dp	
){
    int* sort_keys = reinterpret_cast<int*> (sort_keys_dp);
    int* N_rep = reinterpret_cast<int*> (N_rep_dp);

    sort_N_rep(N, S, sort_keys, N_rep);
}

void reindex_N_rep_python(
	const int N,
	const int S,
	const int D,
	const int G,
	const long N_G_dp,
	const long cc_src_dp,
	const long cc_snk_dp,
	const long G_rep_dp,
	const long G_neuron_counts_dp,
	const long G_group_delay_counts_dp,
    const py::tuple& gc_location,
    const py::tuple& gc_conn_shape,
	long cc_syn_dp,
	long N_delays_dp,
    long sort_keys_dp,
	long N_rep_dp,
	bool verbose = 0
)
{
    const int* N_G = reinterpret_cast<int*> (N_G_dp);
    const int* cc_src = reinterpret_cast<int*> (cc_src_dp);
    const int* cc_snk = reinterpret_cast<int*> (cc_snk_dp);
    const int* G_rep = reinterpret_cast<int*> (G_rep_dp);
    const int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);
    const int* G_group_delay_counts = reinterpret_cast<int*> (G_group_delay_counts_dp);
    int* cc_syn = reinterpret_cast<int*> (cc_syn_dp);
    int* N_delays = reinterpret_cast<int*> (N_delays_dp);
    int* sort_keys = reinterpret_cast<int*> (sort_keys_dp);
    int* N_rep = reinterpret_cast<int*> (N_rep_dp);

    reindex_N_rep(
        N, 
        S, 
        D, 
        G,
        N_G,
        cc_src, 
        cc_snk,
        G_rep, 
        G_neuron_counts, 
        G_group_delay_counts,
        gc_location[0].cast<int>(), gc_location[1].cast<int>(),
        gc_conn_shape[0].cast<int>(), gc_conn_shape[1].cast<int>(),
        cc_syn,
        N_delays,
        sort_keys,
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
          py::arg("curand_states"),
          py::arg("N_G"),
          py::arg("cc_src"),
          py::arg("cc_snk"),
          py::arg("G_rep"),
          py::arg("G_neuron_counts"),
          py::arg("G_group_delay_counts"),
          py::arg("G_autapse_indices"),
          py::arg("G_relative_autapse_indices"),
          py::arg("has_autapses"),
          py::arg("gc_location"),
          py::arg("gc_conn_shape"),
          py::arg("cc_syn"),
          py::arg("N_delays"),
          py::arg("sort_keys"),
          py::arg("N_rep"),
          py::arg("verbose") = false);

    m.def("sort_N_rep", 
          &sort_N_rep_python, 
          py::arg("N"),
          py::arg("S"),
          py::arg("sort_keys"),
          py::arg("N_rep"));
    
    m.def("reindex_N_rep", 
          &reindex_N_rep_python, 
          py::arg("N"),
          py::arg("S"),
          py::arg("D"),
          py::arg("G"),
          py::arg("N_G"),
          py::arg("cc_src"),
          py::arg("cc_snk"),
          py::arg("G_rep"),
          py::arg("G_neuron_counts"),
          py::arg("G_group_delay_counts"),
          py::arg("gc_location"),
          py::arg("gc_conn_shape"),
          py::arg("cc_syn"),
          py::arg("N_delays"),
          py::arg("sort_keys"),
          py::arg("N_rep"),
          py::arg("verbose") = false);

    py::class_<CuRandStates, std::shared_ptr<CuRandStates>>(m, "CuRandStates_") //, py::dynamic_attr())
    .def(py::init<int>())
    .def_readonly("n_states", &CuRandStates::n_states)
    .def_readonly("states", &CuRandStates::states)
    // .def("get_ptr", &CuRandStates::get_ptr)
    // .def_readonly("p", &CuRandStates::p)
    .def("__repr__",
        [](const CuRandStates &cs) {
            return "CuRandStates_(" + std::to_string(cs.n_states) + ")";
        }
    );

    py::class_<CuRandStatesPointer>(m, "CuRandStates") //, py::dynamic_attr())
    .def(py::init<int>())
    .def_property_readonly("n_states", &CuRandStatesPointer::n_states)
    // .def_readonly("states", &CuRandStatesPointer::states)
    // .def("get_ptr", &CuRandStatesPointer::get_ptr)
    .def("ptr", &CuRandStatesPointer::ptr)
    .def("__repr__",
        [](const CuRandStatesPointer &cs) {
            return "CuRandStates(" + std::to_string(cs.ptr_->n_states) + ")";
        }
    );

    m.def("print_random_numbers", 
          &print_random_numbers);

    m.def("print_random_numbers2", 
          &print_random_numbers2);


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
