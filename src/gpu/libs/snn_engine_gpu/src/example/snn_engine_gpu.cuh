// #include <windows.h>
#include <utils/cpp_header.h>
#include <utils/cuda_header.h>

#include <utils/launch_parameters.cuh>


struct CudaGLResource
{
	int* d{ nullptr };  // device pointer

	uint id{};
	size_t size{};

	bool is_mapped = false;

	CudaGLResource();
	CudaGLResource(uint b);

	cudaGraphicsResource* buffer_pt{ nullptr };

	// void register_(uint b);
	// void unregister() const;
	// void unmap();
	void map();
};

int pyadd(int N, long px, long py);

void pyadd_occupancy();