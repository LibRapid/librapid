#include <librapid/internal/config.hpp>
#include <librapid/linalg/blasInterface.hpp>
#include <librapid/modified/modified.hpp>

namespace librapid::blas::impl {
	extended::float16_t dot(int64_t n, extended::float16_t *__restrict x, int64_t incX,
							extended::float16_t *__restrict y, int64_t incY,
							cublasHandle_t *handles) {
#if defined(LIBRAPID_HAS_OMP)
		int64_t threadNum = omp_get_thread_num();
#else
		int64_t threadNum = 0;
#endif
		extended::float16_t result = 0;
		cublasSafeCall(cublasDotEx(handles[threadNum],
								   (int)n,
								   x,
								   CUDA_R_16F,
								   (int)incX,
								   y,
								   CUDA_R_16F,
								   (int)incY,
								   &result,
								   CUDA_R_16F,
								   CUDA_R_32F));

		return result;
	}

	float dot(int64_t n, float *__restrict x, int64_t incX, float *__restrict y, int64_t incY,
			  cublasHandle_t *handles) {
#if defined(LIBRAPID_HAS_OMP)
		int64_t threadNum = omp_get_thread_num();
#else
		int64_t threadNum = 0;
#endif
		float result = 0;
		cublasSafeCall(
		  cublasSdot_v2(handles[threadNum], (int)n, x, (int)incX, y, (int)incY, &result));
		return result;
	}

	double dot(int64_t n, double *__restrict x, int64_t incX, double *__restrict y, int64_t incY,
			   cublasHandle_t *handles) {
#if defined(LIBRAPID_HAS_OMP)
		int64_t threadNum = omp_get_thread_num();
#else
		int64_t threadNum = 0;
#endif
		double result = 0;
		cublasSafeCall(
		  cublasDdot_v2(handles[threadNum], (int)n, x, (int)incX, y, (int)incY, &result));
		return result;
	}
} // namespace librapid::blas::impl