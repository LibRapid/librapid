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

	namespace imp {
		enum class LinAlgOp {
			None,
			VectorVector,
			VectorMatrix,
			MatrixVector,
			MatrixMatrix,
			NDim
		};
	}

	Array Array::dot(const Array &other) const {
#ifdef LIBRAPID_HAS_CUDA
		if (!cublasHandleCreated) {
			cublasHandleCreated = true;
			cublasCreate_v2(&cublasHandle);
			cublasSetStream(cublasHandle, cudaStream);
			cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
		}
#endif // LIBRAPID_HAS_CUDA

		// Find the largest datatype and location
		Datatype resDtype = std::max(m_dtype, other.m_dtype);
		Accelerator resLocn = std::max(m_location, other.m_location);
		Extent resShape;
		bool resIsScalar = false;

		Array lhs;
		Array rhs;

		if (m_location != other.m_location) {
			auto tmpThis = clone(resDtype, resLocn);
			auto tmpOther = clone(resDtype, resLocn);
			// auto tmpThis = clone(resDtype, Accelerator::CPU);
			// auto tmpOther = clone(resDtype, Accelerator::CPU);
			return tmpThis.dot(tmpOther);
		} else {
			imp::LinAlgOp operationType = imp::LinAlgOp::None;
			if (m_extent.ndim() == 1 && other.m_extent.ndim() == 1) {
				// Ensure the arrays have the same number of values
				if (m_extent.size() == other.m_extent.size()) {
					operationType = imp::LinAlgOp::VectorVector; // Vector-Vector
					resShape = Extent({m_extent.size()});
					resIsScalar = true;
				} else {
					throw std::invalid_argument(
							"Cannot compute dot product on vectors with " + m_extent.str() + " and " +
							other.m_extent.str() + ". Both vectors must have the same number of elements");
				}
			} else if (m_extent.ndim() == 1 && other.m_extent.ndim() == 2) {
				// TODO: Figure out what this should do
				operationType = imp::LinAlgOp::VectorMatrix;
				throw std::runtime_error("Vector-matrix products are not implemented yet");
			} else if (m_extent.ndim() == 2 && other.m_extent.ndim() == 1) {
				// Matrix-vector product. Ensure the columns of *this equal the number
				// of elements in other
				if (m_extent[1] == other.m_extent[0]) {
					operationType = imp::LinAlgOp::MatrixVector;
					resShape = Extent({m_extent[0]});
				} else {
					throw std::invalid_argument(
							"Cannot compute dot product on matrix and vector with " + m_extent.str() + " and " +
							other.m_extent.str() + ". Columns of matrix must equal number of elements in vector");
				}
			} else if (m_extent.ndim() == 2 && other.m_extent.ndim() == 2) {
				// Matrix product -- Ensure the columns of *this equal the rows of other
				if (m_extent[1] == other.m_extent[0]) {
					operationType = imp::LinAlgOp::MatrixMatrix;
					resShape = Extent({m_extent[0], other.m_extent[1]});
				} else {
					throw std::invalid_argument(
							"Cannot compute matrix product on matrices with " + m_extent.str() + " and " +
							other.m_extent.str() + ". Columns of first matrix must match rows of second matrix");
				}
			} else if (m_extent.ndim() == other.m_extent.ndim() && m_extent.ndim() > 2) {
				operationType = imp::LinAlgOp::NDim;
				throw std::runtime_error("N-dimensional products are not implemented yet");
			} else {
				throw std::runtime_error(
						"Unknown array product configuration with " + m_extent.str() + " and " + other.m_extent.str() +
						". Valid operation types are: vector-vector product, vector-matrix product, " +
						"matrix-vector product, matrix-matrix product, ndim product");
			}

			Array res(resShape, resDtype, resLocn);
			res.m_isScalar = resIsScalar;

			// TODO: Optimise this to avoid unnecessary cloning
			lhs = (m_stride.isTrivial() && m_stride.isContiguous()) ? *this : clone();
			rhs = (other.m_stride.isTrivial() && other.m_stride.isContiguous()) ? other : other.clone();

			switch (operationType) {
				case imp::LinAlgOp::VectorVector : {
					// Vector dot product
					std::visit([&](auto *c, auto *a, auto *b) {
						using A = typename std::remove_pointer<decltype(a)>::type;
						using B = typename std::remove_pointer<decltype(b)>::type;
						using C = typename std::remove_pointer<decltype(c)>::type;

						int64_t N = m_extent[0];
						int64_t incA = m_stride[0];
						int64_t incB = other.m_stride[0];

						if (resLocn == Accelerator::CPU) {
							*c = (C) linalg::cblas_dot(N, a, incA, b, incB);
						}
								#ifdef LIBRAPID_HAS_CUDA
						else {
							// res = (C) linalg::cblas_dot_cuda(cublasHandle, N, a, incA, b, incB);
							linalg::cblas_dot_cuda(cublasHandle, N, a, incA, b, incB, c);
						}
						#else
						else {
							throw std::runtime_error("CUDA support was not enabled");
						}
						#endif
					}, res.m_dataStart, lhs.m_dataStart, rhs.m_dataStart);
					break;
				}
                case imp::LinAlgOp::MatrixVector : {
                    // Matrix-vector multiplication
                    std::visit([&](auto *c, auto *a, auto *b) {
                        throw std::runtime_error("Not implemented yet");
                    }, res.m_dataStart, lhs.m_dataStart, rhs.m_dataStart);
                    break;
                }
				case imp::LinAlgOp::MatrixMatrix : {
					// 2D array -- matrix matrix multiplication
					std::visit([&](auto *c, auto *a, auto *b) {
						using A = typename std::remove_pointer<decltype(a)>::type;
						using B = typename std::remove_pointer<decltype(b)>::type;
						using C = typename std::remove_pointer<decltype(c)>::type;

						int64_t M = m_extent[0];
						int64_t N = other.m_extent[1];
						int64_t K = m_extent[1];

						bool transA = !m_stride.isTrivial();
						bool transB = !other.m_stride.isTrivial();

						int64_t lda = transA ? M : K;
						int64_t ldb = transB ? K : N;
						int64_t ldc = transB ? M : N;

						A alpha = 1.0;
						C beta = 0.0;

						if (resLocn == Accelerator::CPU) {
							linalg::cblas_gemm('r', transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c,
											   ldc);
						}
								#ifdef LIBRAPID_HAS_CUDA
						else {
							linalg::cblas_gemm_cuda(cublasHandle, transA, transB, M, N, K, alpha, a, lda, b,
													ldb, beta, c, ldc);
						}
						#else
						else {
							throw std::runtime_error("CUDA support was not enabled");
						}
						#endif
					}, res.m_dataStart, lhs.m_dataStart, rhs.m_dataStart);
					break;
				}
			}

			return res;
		}
	}
}