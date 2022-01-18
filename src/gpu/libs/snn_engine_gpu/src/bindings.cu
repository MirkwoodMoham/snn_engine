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


void init_pos_gpu_python(
    int N, 
    int G, 
    long N_pos_dp,
    long N_G_dp,
    const py::tuple& G_shape
){
    float *N_pos = reinterpret_cast<float*> (N_pos_dp);
	int *N_G = reinterpret_cast<int*> (N_G_dp);

    init_pos_gpu(
        N, 
        G, 
        N_pos, 
        N_G, 
        G_shape[0].cast<int>(), 
        G_shape[1].cast<int>(), 
        G_shape[2].cast<int>());
}



PYBIND11_MODULE(snn_engine_gpu, m)
{
    m.def("pyadd", &pyadd, "A function which adds two numbers");
    m.def("print_dict", &print_dict);
    m.def("pyadd_occupancy", &pyadd_occupancy);
    m.def("init_pos_gpu", &init_pos_gpu_python);
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
