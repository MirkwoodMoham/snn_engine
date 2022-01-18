#include <example/snn_engine_gpu.cuh>


CudaGLResource::CudaGLResource(){
}

// CudaGLResource::CudaGLResource(uint b){
//    register_(b);
// }

// void CudaGLResource::register_(uint b)
// {
// 	id = b;
// 	int aa = 0;
// 	int bb = 0;
// 	uint cc = 0;
// 	uint dd = 0;
// 	int* driverVersion = &aa;
// 	int* runtimeVersion = &bb;
// 	uint* ddd = &dd;

//  	cudaDriverGetVersion(driverVersion);
//  	cudaRuntimeGetVersion(runtimeVersion);

//     std::cout << cudaGetLastError() << "\n";

//     std::cout << aa << "\n";
//     std::cout << bb << "\n";
//     std::cout << id << "\n";
//     // cudaGLGetDevices(ddd, aa, cc, cudaGLDeviceListAll);

//     std::cout << aa << "\n";
//     std::cout << cc << "\n";
//     std::cout << ddd << "\n";
// 	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&buffer_pt, id, cudaGraphicsRegisterFlagsNone));
// }

// void CudaGLResource::unregister() const
// {
// 	checkCudaErrors(cudaGraphicsUnregisterResource(buffer_pt));
// }

// void CudaGLResource::unmap()
// {
// 	checkCudaErrors(cudaGraphicsUnmapResources(1, &buffer_pt, nullptr));
// 	is_mapped = false;
// }


void CudaGLResource::map()
{
	if (!is_mapped)
	{
		checkCudaErrors(cudaGraphicsMapResources(1, &buffer_pt, nullptr));
		is_mapped = true;
	}

	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d), &size, buffer_pt));
}

// Error Checking Function
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Simple CUDA kernel
__global__
void cuadd(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + 9 * y[i];
}

// Simple wrapper function to be exposed to Python
int pyadd(int N, long px, long py)
{

	std::cout << "pyadd\n";

	float *x = reinterpret_cast<float*> (px);
	float *y = reinterpret_cast<float*> (py);

	// Run kernel on 1M elements on the GPU
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	cuadd<<<numBlocks, blockSize>>>(N,x,y);
	
	// Wait for GPU to finish before accessing on host
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

  	return 0;
}

void pyadd_occupancy(){
	print_theoretical_occupancy(32, (void *)cuadd);
	//highlighted_print(32);
}

