#include <librapid/config.hpp>
#include <librapid/array/multiarray.hpp>
#include <librapid/utils/array_utils.hpp>
#include <librapid/autocast/autocast.hpp>
#include <librapid/array/cblas_api.hpp>

#include <type_traits>

namespace librapid {
	#ifdef LIBRAPID_HAS_CUDA
	static cublasHandle_t cublasHandle;
	static bool cublasHandleCreated = false;
	#endif // LIBRAPID_HAS_CUDA

	Array Array::dot(const Array &other) const {
		if (!cublasHandleCreated) {
			cublasHandleCreated = true;
			cublasCreate_v2(&cublasHandle);
		}

		// Find the largest datatype and location
		Datatype resDtype = std::max(m_dtype, other.m_dtype);
		Accelerator resLocn = std::max(m_location, other.m_location);

		if (m_location != other.m_location) {
			// auto tmpThis = clone(resDtype, resLocn);
			// auto tmpOther = clone(resDtype, resLocn);
			auto tmpThis = clone(resDtype, Accelerator::CPU);
			auto tmpOther = clone(resDtype, Accelerator::CPU);
			return tmpThis.dot(tmpOther);
		} else {
			// 2D array -- matrix matrix multiplication
			Array res(Extent{m_extent[0], other.m_extent[1]}, resDtype, resLocn);
			const auto &srcA = (m_stride.isTrivial() && m_stride.isContiguous()) ? *this : clone();
			const auto &srcB = (other.m_stride.isTrivial() && other.m_stride.isContiguous()) ? other : other.clone();
			std::visit([&](auto *c, auto *a, auto *b) {
				using A = typename std::remove_pointer<decltype(a)>::type;
				using B = typename std::remove_pointer<decltype(b)>::type;
				using C = typename std::remove_pointer<decltype(c)>::type;

				int64_t M = m_extent[0];
				int64_t N = m_extent[1];
				int64_t K = other.m_extent[1];

				bool transA = !m_stride.isTrivial();
				bool transB = !other.m_stride.isTrivial();

				int64_t lda = transA ? M : K;
				int64_t ldb = transB ? K : N;
				int64_t ldc = transB ? M : N;

				A alpha = 1.0;
				C beta = 0.0;

				linalg::cblas_gemm('r', transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc);
			}, res.m_dataStart, srcA.m_dataStart, srcB.m_dataStart);

			return res;
		}
	}
}