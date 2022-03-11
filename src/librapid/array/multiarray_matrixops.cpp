#include <librapid/array/cblas_api.hpp>
#include <librapid/array/multiarray.hpp>
#include <librapid/autocast/autocast.hpp>
#include <librapid/config.hpp>
#include <librapid/utils/array_utils.hpp>
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

	void dot(const Array &lhs, const Array &rhs, Array &res) {
#ifdef LIBRAPID_HAS_CUDA
		if (!cublasHandleCreated) {
			cublasHandleCreated = true;
			cublasCreate_v2(&cublasHandle);
			cublasSetStream(cublasHandle, cudaStream);
			cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
		}
#endif // LIBRAPID_HAS_CUDA

		// If the result array is not trivial, nothing below will work
		// (currently)
		if (!res.stride().isTrivial() || !res.stride().isContiguous()) {
			throw std::invalid_argument(
			  "Cannot store result of matrix product in non-trivial result "
			  "array");
		}

		// Find the largest datatype and location
		Datatype resDtype	= res.dtype();
		Accelerator resLocn = res.location();

		if (lhs.location() != resLocn || rhs.location() != resLocn) {
			auto tmpLhs = lhs.clone(resDtype, resLocn);
			auto tmpRhs = rhs.clone(resDtype, resLocn);
			dot(tmpLhs, tmpRhs, res);
		} else {
			imp::LinAlgOp operationType = imp::LinAlgOp::None;
			if (lhs.ndim() == 1 && rhs.ndim() == 1) {
				// Ensure the arrays have the same number of values
				if (lhs.extent().size() == rhs.extent().size() &&
					res.isScalar()) {
					operationType =
					  imp::LinAlgOp::VectorVector; // Vector-Vector
				} else {
					throw std::invalid_argument(
					  "Cannot compute dot product on vectors with " +
					  lhs.extent().str() + " and " + rhs.extent().str() +
					  " and store the result in Array with " +
					  res.extent().str() +
					  ". Both vectors must have the same number of elements "
					  "and the "
					  "result array must be a scalar");
				}
			} else if (lhs.ndim() == 1 && rhs.ndim() == 2) {
				// TODO: Figure out what this should do
				operationType = imp::LinAlgOp::VectorMatrix;
				throw std::runtime_error(
				  "Vector-matrix products are not implemented yet");
			} else if (lhs.ndim() == 2 && rhs.ndim() == 1) {
				// Matrix-vector product. Ensure the columns of *this equal the
				// number of elements in other
				if (lhs.extent()[1] == rhs.extent()[0] &&
					lhs.extent()[0] == res.extent()[0]) {
					operationType = imp::LinAlgOp::MatrixVector;
				} else {
					throw std::invalid_argument(
					  "Cannot compute dot product on matrix and vector with " +
					  lhs.extent().str() + " and " + rhs.extent().str() +
					  " and store result in array with " + res.extent().str() +
					  ". The columns of the matrix must equal number of "
					  "elements in the "
					  "vector, and the number " +
					  "of elements in the result vector must match the rows of "
					  "the "
					  "matrix");
				}
			} else if (lhs.ndim() == 2 && rhs.ndim() == 2) {
				// Matrix product -- Ensure the columns of *this equal the rows
				// of other
				if (lhs.extent()[1] == rhs.extent()[0] &&
					res.extent()[0] == lhs.extent()[0] &&
					res.extent()[1] == rhs.extent()[1]) {
					operationType = imp::LinAlgOp::MatrixMatrix;
				} else {
					throw std::invalid_argument(
					  "Cannot compute matrix product on matrices with " +
					  lhs.extent().str() + " and " + rhs.extent().str() +
					  " and store the result in an array with " +
					  res.extent().str() +
					  ". The columns of first matrix must match the rows of "
					  "second "
					  "matrix, and the result matrix" +
					  " must have extent {firstMatrix.rows, "
					  "secondMatrix.cols}");
				}
			} else if (lhs.ndim() == rhs.ndim() && lhs.ndim() > 2) {
				operationType = imp::LinAlgOp::NDim;
				throw std::runtime_error(
				  "N-dimensional products are not implemented yet");
			} else {
				throw std::runtime_error(
				  "Unknown array product configuration with " +
				  lhs.extent().str() + " and " + rhs.extent().str() +
				  ". Valid operation types are: vector-vector product, " +
				  "vector-matrix product, matrix-vector product, matrix-matrix "
				  "product, ndim product");
			}

			// TODO: Optimise this to avoid unnecessary cloning
			auto lhsTemp =
			  (lhs.stride().isTrivial() && lhs.stride().isContiguous())
				? lhs
				: lhs.clone();
			auto rhsTemp =
			  (rhs.stride().isTrivial() && rhs.stride().isContiguous())
				? rhs
				: rhs.clone();

			switch (operationType) {
				case imp::LinAlgOp::VectorVector: {
					// Vector dot product
					std::visit(
					  [&](auto *c, auto *a, auto *b) {
						  using A =
							typename std::remove_pointer<decltype(a)>::type;
						  using B =
							typename std::remove_pointer<decltype(b)>::type;
						  using C =
							typename std::remove_pointer<decltype(c)>::type;

						  int64_t N	   = lhs.extent()[0];
						  int64_t incA = lhs.stride()[0];
						  int64_t incB = rhs.stride()[0];

						  if (resLocn == Accelerator::CPU) {
							  *c = (C)linalg::cblas_dot(N, a, incA, b, incB);
						  }
#ifdef LIBRAPID_HAS_CUDA
						  else {
							  // res = (C) linalg::cblas_dot_cuda(cublasHandle,
							  // N, a, incA, b, incB);
							  linalg::cblas_dot_cuda(
								cublasHandle, N, a, incA, b, incB, c);
						  }
#else
						  else {
							  throw std::runtime_error(
								"CUDA support was not enabled");
						  }
#endif
					  },
					  res._dataStart(),
					  lhs._dataStart(),
					  rhs._dataStart());
					break;
				}
				case imp::LinAlgOp::MatrixVector: {
					// Matrix-vector multiplication
					std::visit(
					  [&](auto *c, auto *a, auto *b) {
						  using A =
							typename std::remove_pointer<decltype(a)>::type;
						  using B =
							typename std::remove_pointer<decltype(b)>::type;
						  using C =
							typename std::remove_pointer<decltype(c)>::type;

						  bool trans = !lhs.stride().isTrivial();
						  int64_t M	 = lhs.extent()[0];
						  int64_t N	 = rhs.extent()[0];
						  int64_t K	 = lhs.extent()[1];

						  int64_t incX = rhs.stride()[0];
						  int64_t incY = res.stride()[0];

						  A alpha = 1.0;
						  C beta  = 0.0;

						  int64_t lda = trans ? M : K;

						  if (resLocn == Accelerator::CPU) {
							  linalg::cblas_gemv('r',
												 trans,
												 M,
												 N,
												 alpha,
												 a,
												 lda,
												 b,
												 incX,
												 beta,
												 c,
												 incY);
						  }
#ifdef LIBRAPID_HAS_CUDA
						  else {
							  linalg::cblas_gemv_cuda(cublasHandle,
													  'r',
													  trans,
													  M,
													  N,
													  alpha,
													  a,
													  lda,
													  b,
													  incX,
													  beta,
													  c,
													  incY);
						  }
#else
						  else {
							  throw std::runtime_error(
								"CUDA support was not enabled");
						  }
#endif
					  },
					  res._dataStart(),
					  lhs._dataStart(),
					  rhs._dataStart());
					break;
				}
				case imp::LinAlgOp::MatrixMatrix: {
					// 2D array -- matrix matrix multiplication
					std::visit(
					  [&](auto *c, auto *a, auto *b) {
						  using A =
							typename std::remove_pointer<decltype(a)>::type;
						  using B =
							typename std::remove_pointer<decltype(b)>::type;
						  using C =
							typename std::remove_pointer<decltype(c)>::type;

						  int64_t M = lhs.extent()[0];
						  int64_t N = rhs.extent()[1];
						  int64_t K = lhs.extent()[1];

						  bool transA = !lhs.stride().isTrivial();
						  bool transB = !rhs.stride().isTrivial();

						  int64_t lda = transA ? M : K;
						  int64_t ldb = transB ? K : N;
						  int64_t ldc = transB ? M : N;

						  A alpha = 1.0;
						  C beta  = 0.0;

						  if (resLocn == Accelerator::CPU) {
							  linalg::cblas_gemm('r',
												 transA,
												 transB,
												 M,
												 N,
												 K,
												 alpha,
												 a,
												 lda,
												 b,
												 ldb,
												 beta,
												 c,
												 ldc);
						  }
#ifdef LIBRAPID_HAS_CUDA
						  else {
							  linalg::cblas_gemm_cuda(cublasHandle,
													  transA,
													  transB,
													  M,
													  N,
													  K,
													  alpha,
													  a,
													  lda,
													  b,
													  ldb,
													  beta,
													  c,
													  ldc);
						  }
#else
						  else {
							  throw std::runtime_error(
								"CUDA support was not enabled");
						  }
#endif
					  },
					  res._dataStart(),
					  lhs._dataStart(),
					  rhs._dataStart());
					break;
				}
			}
		}
	}

	Array dot(const Array &lhs, const Array &rhs) { return lhs.dot(rhs); }

	Array Array::dot(const Array &other) const {
		// Find the largest datatype and location
		Datatype resDtype	= std::max(m_dtype, other.m_dtype);
		Accelerator resLocn = std::max(m_location, other.m_location);
		Extent resShape;
		bool resIsScalar = false;

		if (m_extent.ndim() == 1 && other.m_extent.ndim() == 1) {
			// Ensure the arrays have the same number of values
			resShape	= Extent({m_extent.size()});
			resIsScalar = true;
		} else if (m_extent.ndim() == 1 && other.m_extent.ndim() == 2) {
			// TODO: Figure out what this should do
			throw std::runtime_error(
			  "Vector-matrix products are not implemented yet");
		} else if (m_extent.ndim() == 2 && other.m_extent.ndim() == 1) {
			// Matrix-vector product. Ensure the columns of *this equal the
			// number of elements in other
			resShape = Extent({m_extent[0]});
		} else if (m_extent.ndim() == 2 && other.m_extent.ndim() == 2) {
			// Matrix product -- Ensure the columns of *this equal the rows of
			// other
			resShape = Extent({m_extent[0], other.m_extent[1]});
		} else if (m_extent.ndim() == other.m_extent.ndim() &&
				   m_extent.ndim() > 2) {
			throw std::runtime_error(
			  "N-dimensional products are not implemented yet");
		}

		Array res(resShape, resDtype, resLocn);
		res.m_isScalar = resIsScalar;

		librapid::dot(*this, other, res);

		return res;
	}
} // namespace librapid