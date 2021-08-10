#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>
#include <stdexcept>
#include <chrono>

#include <librapid/config.hpp>
#include <librapid/cuda/backend.cuh>

static const char *getCublasErrorEnum_(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "UNKNOWN ERROR";
}

inline void cublasSafeCall_(cublasStatus_t err, const char *file, const int line)
{
    if (CUBLAS_STATUS_SUCCESS != err)
		throw std::runtime_error("cuBLAS function failed at line "
		+ std::to_string(line) + ", file " + std::string(file) + ": " +
		getCublasErrorEnum_(err));
}

inline void cudaSafeCall_(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
	throw std::runtime_error("CUDA function failed at line "
	+ std::to_string(line) + ", file " + std::string(file) + ": " + 
	cudaGetErrorString(err));
}

__global__ void print_something_kernel(int x)
{
	printf("Here is something: %i\n", x);
}

template<typename A, typename B, typename C>
__global__ void librapid_cuda_trivial_binary(const A *a, const B *b, C *c,
											 const lr_int stride_a,
											 const lr_int stride_b,
											 const lr_int stride_c,
											 LAMBDA op)
{

}

void print_something(int x)
{
	print_something_kernel<<<1, 1>>>(x);
	cudaDeviceSynchronize();
}

void *librapid_cuda_malloc(size_t bytes)
{
	void *res;
	cudaSafeCall(cudaMalloc(&res, bytes));
	return res;
}

void librapid_cuda_free(void *data)
{
	cudaSafeCall(cudaFree(data));
}

void librapid_cuda_memcpy(void *dst, void *src, size_t bytes, librapid_cuda_memcpykind kind)
{
	cudaSafeCall(cudaMemcpy(dst, src, bytes, (cudaMemcpyKind) kind));
}

template<typename A, typename B, typenaem C>
void librapid_cuda_binary_op(const A *a, const B *b, C *c,
							 bool trivial_a, bool trivial_b, bool trivial_c,
							 const lr_int[LIBRAPID_MAX_DIMS] stride_a,
							 const lr_int[LIBRAPID_MAX_DIMS] stride_b,
							 const lr_int[LIBRAPID_MAX_DIMS] stride_c)
{
	cudaSafeCall(cudaDeviceSynchronize());

	// Check for simple strides
	if (trivial_a && trivial_b && trivial_c)
	{
		// Use vector add

	}
}
