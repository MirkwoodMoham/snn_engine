#include <pybind11/include/pybind11/pybind11.h>
#include <pybind11/include/pybind11/numpy.h>
#include <pybind11/include/pybind11/stl.h>

#include <example/snn_engine_gpu.cuh>
#include <utils/launch_parameters.cuh>
#include <snn_construction.cuh>


namespace py = pybind11;


void print_dict(const py::dict& dict) {
    /* Easily interact with Python types */
    for (auto item : dict)
        std::cout << "key=" << std::string(py::str(item.first)) << ", "
                  << "value=" << std::string(py::str(item.second)) << std::endl;
}

void print_3tuple(const py::tuple& tuple) {
    /* Easily interact with Python types */
    
    std::cout << "(" << std::string(py::str(tuple[0]))
        << ", " << std::string(py::str(tuple[1])) 
        << ", " << std::string(py::str(tuple[2])) 
        << ")"
        << std::endl;
}


void set_G_info_python(
    const int N, 
    const int G, 
    long N_pos_dp,
    const py::tuple& N_pos_shape,
    long N_G_dp,
    const py::tuple& G_shape,
    long G_neuron_counts_dp,
    const int N_pos_n_cols,
    const int N_G_n_cols,
	const int N_G_neuron_type_col,
	const int N_G_group_id_col
){
    float* N_pos = reinterpret_cast<float*> (N_pos_dp);
	int* N_G = reinterpret_cast<int*> (N_G_dp);
	int* G_neuron_counts = reinterpret_cast<int*> (G_neuron_counts_dp);

    set_G_info(
        N, 
        G, 
        N_pos, 
        N_G, 
        G_neuron_counts,
        G_shape[0].cast<int>(), G_shape[1].cast<int>(), G_shape[2].cast<int>(),
        N_pos_shape[0].cast<float>(), N_pos_shape[1].cast<float>(), N_pos_shape[2].cast<float>(),
        N_pos_n_cols,
        N_G_n_cols,
        N_G_neuron_type_col,
        N_G_group_id_col
    );
}



PYBIND11_MODULE(snn_engine_gpu, m)
{
    m.def("set_G_info", 
          &set_G_info_python, 
          py::arg("N"),
          py::arg("G"),
          py::arg("N_pos_dp"),
          py::arg("N_pos_shape"),
          py::arg("N_G_dp"),
          py::arg("G_shape"),
          py::arg("G_neuron_counts_dp"),
          py::arg("N_pos_n_cols") = 13, 
          py::arg("N_G_n_cols") = 3,
          py::arg("N_G_neuron_type_col") = 0,
          py::arg("N_G_group_id_col") = 1
    );
    
    m.def("pyadd", &pyadd, "A function which adds two numbers");
    m.def("print_dict", &print_dict);
    m.def("pyadd_occupancy", &pyadd_occupancy);
    
    py::class_<CudaGLResource>(m, "CudaGLResource", py::dynamic_attr())
        .def(py::init())
        .def("map", &CudaGLResource::map)
        // .def("unmap", &CudaGLResource::unmap)
        // .def("register", &CudaGLResource::register_)
        // .def("unregister", &CudaGLResource::unregister)
        .def_readonly("id", &CudaGLResource::id)
        .def_readonly("size", &CudaGLResource::size)
        .def_readonly("is_mapped", &CudaGLResource::is_mapped)
        .def("__repr__",
            [](const CudaGLResource &a) {
                return "<CudaGLResource(id=" + std::to_string(a.id) + ",is_mapped=" + std::to_string(a.is_mapped) + ")>";
            }
    );
}
