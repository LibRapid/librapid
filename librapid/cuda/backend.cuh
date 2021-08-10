#ifndef LIBRAPID_CUDA_BACKEND
#define LIBRAPID_CUDA_BACKEND

#include <string>

#ifdef LIBRAPID_CUDA
#define LIBRAPID_DLL __declspec(dllimport)
#else
#define LIBRAPID_DLL __declspec(dllexport)
#endif

#ifndef __NVCC__
enum cublasStatus_t { A_CUBLAS_STATUS };
enum cudaStatus_t { A_CUDA_STATUS };
enum cudaError_t { A_CUDA_ERROR };
#endif

// cuBLAS API errors
static const char *getCublasErrorEnum_(cublasStatus_t error);

//********************//
// cuBLAS ERROR CHECK //
//********************//
#ifndef cublasSafeCall
#define cublasSafeCall(err)     cublasSafeCall_(err, __FILE__, __LINE__)
#endif

inline void cublasSafeCall_(cublasStatus_t err, const char *file, const int line);

//********************//
//  CUDA ERROR CHECK  //
//********************//
#ifndef cudaSafeCall
#define cudaSafeCall(err)     cudaSafeCall_(err, __FILE__, __LINE__)
#endif

inline void cudaSafeCall_(cudaError_t err, const char *file, const int line);

// Enums
enum class librapid_cuda_memcpykind
{
	HOST_HOST = 0,
	HOST_DEVICE = 1,
	DEVICE_HOST = 2,
	DEVICE_DEVICE = 3
};

#ifdef __cplusplus
extern "C" {
#endif

// Expose these functions to librapid

LIBRAPID_DLL void print_something(int x);
LIBRAPID_DLL void *librapid_cuda_malloc(size_t bytes);
LIBRAPID_DLL void librapid_cuda_free(void *data);
LIBRAPID_DLL void librapid_cuda_memcpy(void *dst, void *src, size_t bytes, librapid_cuda_memcpykind kind);
LIBRAPID_DLL void librapid_cuda_malloc_test(size_t bytes);

#ifdef __cplusplus
}
#endif

#endif // LIBRAPID_CUDA_BACKEND