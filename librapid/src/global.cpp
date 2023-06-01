#include <librapid/librapid.hpp>

namespace librapid::global {
	bool throwOnAssert				 = false;
	size_t multithreadThreshold	 = 5000;
	size_t gemmMultithreadThreshold = 100;
	size_t numThreads				 = 8;
	size_t cacheLineSize			 = 64;
	size_t memoryAlignment			 = LIBRAPID_DEFAULT_MEM_ALIGN;

#if defined(LIBRAPID_HAS_OPENCL)
	std::vector<cl::Device> openclDevices;
	cl::Context openCLContext;
	cl::Device openCLDevice;
	cl::CommandQueue openCLQueue;
	cl::Program::Sources openCLSources;
	cl::Program openCLProgram;
	bool openCLConfigured = false;
#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)
	cudaStream_t cudaStream;
	cublasHandle_t cublasHandle;
	jitify::JitCache jitCache;
#endif // LIBRAPID_HAS_CUDA
} // namespace librapid::global
