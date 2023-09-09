#ifndef LIBRAPID_ARRAY_TRANSPOSE_HPP
#define LIBRAPID_ARRAY_TRANSPOSE_HPP

namespace librapid {
	namespace typetraits {
		template<typename T>
		struct TypeInfo<array::Transpose<T>> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::ArrayFunction;
			using Scalar							   = typename TypeInfo<std::decay_t<T>>::Scalar;
			using Backend	  = typename TypeInfo<std::decay_t<T>>::Backend;
			using ShapeType	  = typename TypeInfo<std::decay_t<T>>::ShapeType;
			using StorageType = typename TypeInfo<std::decay_t<T>>::StorageType;
			static constexpr bool allowVectorisation = false;
		};

		LIBRAPID_DEFINE_AS_TYPE(typename T, array::Transpose<T>);
	} // namespace typetraits

	namespace kernels {
#if defined(LIBRAPID_NATIVE_ARCH)
#	if !defined(LIBRAPID_APPLE) && LIBRAPID_ARCH >= ARCH_AVX2
#		define LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE 4
#		define LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE 8

		template<typename Alpha>
		LIBRAPID_ALWAYS_INLINE void transposeFloatKernel(float *__restrict out,
														 float *__restrict in, Alpha alpha,
														 int64_t cols) {
			__m256 r0, r1, r2, r3, r4, r5, r6, r7;
			__m256 t0, t1, t2, t3, t4, t5, t6, t7;

#		define LOAD256_IMPL(LEFT_, RIGHT_)                                                        \
			_mm256_insertf128_ps(                                                                  \
			  _mm256_castps128_ps256(_mm_loadu_ps(&(LEFT_))), _mm_loadu_ps(&(RIGHT_)), 1)

			r0 = LOAD256_IMPL(in[0 * cols + 0], in[4 * cols + 0]);
			r1 = LOAD256_IMPL(in[1 * cols + 0], in[5 * cols + 0]);
			r2 = LOAD256_IMPL(in[2 * cols + 0], in[6 * cols + 0]);
			r3 = LOAD256_IMPL(in[3 * cols + 0], in[7 * cols + 0]);
			r4 = LOAD256_IMPL(in[0 * cols + 4], in[4 * cols + 4]);
			r5 = LOAD256_IMPL(in[1 * cols + 4], in[5 * cols + 4]);
			r6 = LOAD256_IMPL(in[2 * cols + 4], in[6 * cols + 4]);
			r7 = LOAD256_IMPL(in[3 * cols + 4], in[7 * cols + 4]);

#		undef LOAD256_IMPL

			t0 = _mm256_unpacklo_ps(r0, r1);
			t1 = _mm256_unpackhi_ps(r0, r1);
			t2 = _mm256_unpacklo_ps(r2, r3);
			t3 = _mm256_unpackhi_ps(r2, r3);
			t4 = _mm256_unpacklo_ps(r4, r5);
			t5 = _mm256_unpackhi_ps(r4, r5);
			t6 = _mm256_unpacklo_ps(r6, r7);
			t7 = _mm256_unpackhi_ps(r6, r7);

			__m256 v;

			v  = _mm256_shuffle_ps(t0, t2, 0x4E);
			r0 = _mm256_blend_ps(t0, v, 0xCC);
			r1 = _mm256_blend_ps(t2, v, 0x33);

			v  = _mm256_shuffle_ps(t1, t3, 0x4E);
			r2 = _mm256_blend_ps(t1, v, 0xCC);
			r3 = _mm256_blend_ps(t3, v, 0x33);

			v  = _mm256_shuffle_ps(t4, t6, 0x4E);
			r4 = _mm256_blend_ps(t4, v, 0xCC);
			r5 = _mm256_blend_ps(t6, v, 0x33);

			v  = _mm256_shuffle_ps(t5, t7, 0x4E);
			r6 = _mm256_blend_ps(t5, v, 0xCC);
			r7 = _mm256_blend_ps(t7, v, 0x33);

			__m256 alphaVec = _mm256_set1_ps(alpha);

			// Must store unaligned, since the indices are not guaranteed to be aligned
			_mm256_storeu_ps(&out[0 * cols], _mm256_mul_ps(r0, alphaVec));
			_mm256_storeu_ps(&out[1 * cols], _mm256_mul_ps(r1, alphaVec));
			_mm256_storeu_ps(&out[2 * cols], _mm256_mul_ps(r2, alphaVec));
			_mm256_storeu_ps(&out[3 * cols], _mm256_mul_ps(r3, alphaVec));
			_mm256_storeu_ps(&out[4 * cols], _mm256_mul_ps(r4, alphaVec));
			_mm256_storeu_ps(&out[5 * cols], _mm256_mul_ps(r5, alphaVec));
			_mm256_storeu_ps(&out[6 * cols], _mm256_mul_ps(r6, alphaVec));
			_mm256_storeu_ps(&out[7 * cols], _mm256_mul_ps(r7, alphaVec));
		}

		template<typename Alpha>
		LIBRAPID_ALWAYS_INLINE void transposeDoubleKernel(double *__restrict out,
														  double *__restrict in, Alpha alpha,
														  int64_t cols) {
			__m256d r0, r1, r2, r3;
			__m256d t0, t1, t2, t3;

#		define LOAD256_IMPL(LEFT_, RIGHT_)                                                        \
			_mm256_insertf128_pd(                                                                  \
			  _mm256_castpd128_pd256(_mm_loadu_pd(&(LEFT_))), _mm_loadu_pd(&(RIGHT_)), 1)

			r0 = LOAD256_IMPL(in[0 * cols + 0], in[2 * cols + 0]);
			r1 = LOAD256_IMPL(in[1 * cols + 0], in[3 * cols + 0]);
			r2 = LOAD256_IMPL(in[0 * cols + 2], in[2 * cols + 2]);
			r3 = LOAD256_IMPL(in[1 * cols + 2], in[3 * cols + 2]);

#		undef LOAD256_IMPL

			t0 = _mm256_unpacklo_pd(r0, r1);
			t1 = _mm256_unpackhi_pd(r0, r1);
			t2 = _mm256_unpacklo_pd(r2, r3);
			t3 = _mm256_unpackhi_pd(r2, r3);

			__m256d v;

			v  = _mm256_shuffle_pd(t0, t2, 0x0);
			r0 = _mm256_blend_pd(t0, v, 0xC);
			r1 = _mm256_blend_pd(t2, v, 0x3);

			v  = _mm256_shuffle_pd(t1, t3, 0x0);
			r2 = _mm256_blend_pd(t1, v, 0xC);
			r3 = _mm256_blend_pd(t3, v, 0x3);

			__m256d alphaVec = _mm256_set1_pd(alpha);

			_mm256_store_pd(&out[0 * cols], _mm256_mul_pd(r0, alphaVec));
			_mm256_store_pd(&out[1 * cols], _mm256_mul_pd(r1, alphaVec));
			_mm256_store_pd(&out[2 * cols], _mm256_mul_pd(r2, alphaVec));
			_mm256_store_pd(&out[3 * cols], _mm256_mul_pd(r3, alphaVec));
		}
#	elif !defined(LIBRAPID_APPLE) && LIBRAPID_ARCH >= ARCH_SSE

#		define LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE 2
#		define LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE 4

		template<typename Alpha>
		LIBRAPID_ALWAYS_INLINE void transposeFloatKernel(float *__restrict out,
														 float *__restrict in, Alpha alpha,
														 int64_t cols) {
			__m128 tmp3, tmp2, tmp1, tmp0;

			tmp0 = _mm_shuffle_ps(_mm_loadu_ps(in + 0 * cols), _mm_loadu_ps(in + 1 * cols), 0x44);
			tmp2 = _mm_shuffle_ps(_mm_loadu_ps(in + 0 * cols), _mm_loadu_ps(in + 1 * cols), 0xEE);
			tmp1 = _mm_shuffle_ps(_mm_loadu_ps(in + 2 * cols), _mm_loadu_ps(in + 3 * cols), 0x44);
			tmp3 = _mm_shuffle_ps(_mm_loadu_ps(in + 2 * cols), _mm_loadu_ps(in + 3 * cols), 0xEE);

			__m128 alphaVec = _mm_set1_ps(alpha);

			_mm_storeu_ps(out + 0 * cols, _mm_mul_ps(_mm_shuffle_ps(tmp0, tmp1, 0x88), alphaVec));
			_mm_storeu_ps(out + 1 * cols, _mm_mul_ps(_mm_shuffle_ps(tmp0, tmp1, 0xDD), alphaVec));
			_mm_storeu_ps(out + 2 * cols, _mm_mul_ps(_mm_shuffle_ps(tmp2, tmp3, 0x88), alphaVec));
			_mm_storeu_ps(out + 3 * cols, _mm_mul_ps(_mm_shuffle_ps(tmp2, tmp3, 0xDD), alphaVec));
		}

		template<typename Alpha>
		LIBRAPID_ALWAYS_INLINE void transposeDoubleKernel(double *__restrict out,
														  double *__restrict in, Alpha alpha,
														  int64_t cols) {
			__m128d tmp0, tmp1;

			// Load the values from input matrix
			tmp0 = _mm_loadu_pd(in + 0 * cols);
			tmp1 = _mm_loadu_pd(in + 1 * cols);

			// Transpose the 2x2 matrix
			__m128d tmp0Unpck = _mm_unpacklo_pd(tmp0, tmp1);
			__m128d tmp1Unpck = _mm_unpackhi_pd(tmp0, tmp1);

			// Store the transposed values in the output matrix
			__m128d alphaVec = _mm_set1_pd(alpha);
			_mm_storeu_pd(out + 0 * cols, _mm_mul_pd(tmp0Unpck, alphaVec));
			_mm_storeu_pd(out + 1 * cols, _mm_mul_pd(tmp1Unpck, alphaVec));
		}

#	endif // LIBRAPID_MSVC
#endif	   // LIBRAPID_NATIVE_ARCH

		// Ensure the kernel size is always defined, even if the above code doesn't define it
#ifndef LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE
#	define LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE 0
#endif // LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE

#ifndef LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE
#	define LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE 0
#endif // LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE

	} // namespace kernels

	namespace detail {
		namespace cpu {
			template<typename Scalar, typename Alpha>
			LIBRAPID_ALWAYS_INLINE void
			transposeImpl(Scalar *__restrict out, const Scalar *__restrict in, int64_t rows,
						  int64_t cols, Alpha alpha, int64_t blockSize) {
#if !defined(LIBRAPID_OPTIMISE_SMALL_ARRAYS)
				if (rows * cols > global::multithreadThreshold) {
#	pragma omp parallel for shared(rows, cols, blockSize, in, out, alpha) default(none)           \
	  num_threads((int)global::numThreads)
					for (int64_t i = 0; i < rows; i += blockSize) {
						for (int64_t j = 0; j < cols; j += blockSize) {
							for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
								for (int64_t col = j; col < j + blockSize && col < cols; ++col) {
									out[col * rows + row] = in[row * cols + col] * alpha;
								}
							}
						}
					}
				} else
#endif // LIBRAPID_OPTIMISE_SMALL_ARRAYS
				{
					for (int64_t i = 0; i < rows; i += blockSize) {
						for (int64_t j = 0; j < cols; j += blockSize) {
							for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
								for (int64_t col = j; col < j + blockSize && col < cols; ++col) {
									out[col * rows + row] = in[row * cols + col] * alpha;
								}
							}
						}
					}
				}
			}

#if LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE > 0
			template<typename Alpha>
			LIBRAPID_ALWAYS_INLINE void transposeImpl(float *__restrict out, float *__restrict in,
													  int64_t rows, int64_t cols, Alpha alpha,
													  int64_t) {
				constexpr int64_t blockSize = LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE;

#	if !defined(LIBRAPID_OPTIMISE_SMALL_ARRAYS)
				if (rows * cols > global::multithreadThreshold) {
#		pragma omp parallel for shared(rows, cols, in, out, alpha) default(none)                  \
		  num_threads((int)global::numThreads)
					for (int64_t i = 0; i < rows; i += blockSize) {
						for (int64_t j = 0; j < cols; j += blockSize) {
							if (i + blockSize <= rows && j + blockSize <= cols) {
								kernels::transposeFloatKernel(
								  &out[j * rows + i], &in[i * cols + j], alpha, rows);
							} else {
								for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
									for (int64_t col = j; col < j + blockSize && col < cols;
										 ++col) {
										out[col * rows + row] = in[row * cols + col];
									}
								}
							}
						}
					}
				} else
#	endif
				{
					for (int64_t i = 0; i < rows; i += blockSize) {
						for (int64_t j = 0; j < cols; j += blockSize) {
							if (i + blockSize <= rows && j + blockSize <= cols) {
								kernels::transposeFloatKernel(
								  &out[j * rows + i], &in[i * cols + j], alpha, rows);
							} else {
								for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
									for (int64_t col = j; col < j + blockSize && col < cols;
										 ++col) {
										out[col * rows + row] = in[row * cols + col];
									}
								}
							}
						}
					}
				}
			}
#endif // LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE > 0

#if LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE > 0
			template<typename Alpha>
			LIBRAPID_ALWAYS_INLINE void transposeImpl(double *__restrict out, double *__restrict in,
													  int64_t rows, int64_t cols, Alpha alpha,
													  int64_t) {
				constexpr int64_t blockSize = LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE;

#	if !defined(LIBRAPID_OPTIMISE_SMALL_ARRAYS)
				if (rows * cols > global::multithreadThreshold) {
#		pragma omp parallel for shared(rows, cols, in, out, alpha) default(none)                  \
		  num_threads((int)global::numThreads)
					for (int64_t i = 0; i < rows; i += blockSize) {
						for (int64_t j = 0; j < cols; j += blockSize) {
							if (i + blockSize <= rows && j + blockSize <= cols) {
								kernels::transposeDoubleKernel(
								  &out[j * rows + i], &in[i * cols + j], alpha, rows);
							} else {
								for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
									for (int64_t col = j; col < j + blockSize && col < cols;
										 ++col) {
										out[col * rows + row] = in[row * cols + col] * alpha;
									}
								}
							}
						}
					}
				} else
#	endif // LIBRAPID_OPTIMISE_SMALL_ARRAYS
				{
					for (int64_t i = 0; i < rows; i += blockSize) {
						for (int64_t j = 0; j < cols; j += blockSize) {
							if (i + blockSize <= rows && j + blockSize <= cols) {
								kernels::transposeDoubleKernel(
								  &out[j * rows + i], &in[i * cols + j], alpha, rows);
							} else {
								for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
									for (int64_t col = j; col < j + blockSize && col < cols;
										 ++col) {
										out[col * rows + row] = in[row * cols + col] * alpha;
									}
								}
							}
						}
					}
				}
			}
#endif	  // LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE > 0
		} // namespace cpu

#if defined(LIBRAPID_HAS_OPENCL)

		namespace opencl {
			template<typename Scalar, typename Alpha>
			LIBRAPID_ALWAYS_INLINE void transposeImpl(cl::Buffer &out, const cl::Buffer &in,
													  int64_t rows, int64_t cols, Alpha alpha,
													  int64_t) {
				std::string kernelName =
				  fmt::format("transpose_{}", typetraits::TypeInfo<Scalar>::name);
				cl::Kernel kernel(global::openCLProgram, kernelName.c_str());
				kernel.setArg(0, out);
				kernel.setArg(1, in);
				kernel.setArg(2, int(rows));
				kernel.setArg(3, int(cols));
				kernel.setArg(4, Scalar(alpha));
				int TILE_DIM = 16;
				cl::NDRange global((cols + TILE_DIM - 1) / TILE_DIM * TILE_DIM,
								   (rows + TILE_DIM - 1) / TILE_DIM * TILE_DIM);
				cl::NDRange local(TILE_DIM, TILE_DIM);
				auto ret =
				  global::openCLQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
				LIBRAPID_ASSERT(ret == CL_SUCCESS, "OpenCL kernel failed");
			}
		} // namespace opencl

#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)
		namespace cuda {
			template<typename Scalar, typename Alpha>
			LIBRAPID_ALWAYS_INLINE void transposeImpl(Scalar *__restrict out, Scalar *__restrict in,
													  int64_t rows, int64_t cols, Alpha alpha,
													  int64_t blockSize) {
				LIBRAPID_NOT_IMPLEMENTED
			}

			template<typename Alpha>
			LIBRAPID_ALWAYS_INLINE void transposeImpl(float *__restrict out, float *__restrict in,
													  int64_t rows, int64_t cols, Alpha alpha,
													  int64_t) {
				float zero = 0.0f;
				cublasSafeCall(cublasSgeam(global::cublasHandle,
										   CUBLAS_OP_T,
										   CUBLAS_OP_N,
										   (int)rows,
										   (int)cols,
										   &alpha,
										   in,
										   (int)cols,
										   &zero,
										   in,
										   (int)cols,
										   out,
										   rows));
			}

			template<typename Alpha>
			LIBRAPID_ALWAYS_INLINE void transposeImpl(double *__restrict out, double *__restrict in,
													  int64_t rows, int64_t cols, Alpha alpha,
													  int64_t) {
				double zero = 0.0;
				cublasSafeCall(cublasDgeam(global::cublasHandle,
										   CUBLAS_OP_T,
										   CUBLAS_OP_N,
										   rows,
										   cols,
										   &alpha,
										   in,
										   cols,
										   &zero,
										   in,
										   cols,
										   out,
										   rows));
			}

			template<typename Alpha>
			LIBRAPID_ALWAYS_INLINE void transposeImpl(Complex<float> *__restrict out,
													  Complex<float> *__restrict in, int64_t rows,
													  int64_t cols, Complex<Alpha> alpha, int64_t) {
				cuComplex alphaCu {alpha.real(), alpha.imag()};
				cuComplex zero {0.0f, 0.0f};
				cublasSafeCall(cublasCgeam(global::cublasHandle,
										   CUBLAS_OP_T,
										   CUBLAS_OP_N,
										   rows,
										   cols,
										   &alphaCu,
										   reinterpret_cast<cuComplex *>(in),
										   cols,
										   &zero,
										   reinterpret_cast<cuComplex *>(in),
										   cols,
										   reinterpret_cast<cuComplex *>(out),
										   rows));
			}

			template<typename Alpha>
			LIBRAPID_ALWAYS_INLINE void transposeImpl(Complex<double> *__restrict out,
													  Complex<double> *__restrict in, int64_t rows,
													  int64_t cols, Complex<Alpha> alpha, int64_t) {
				cuDoubleComplex alphaCu {alpha.real(), alpha.imag()};
				cuDoubleComplex zero {0.0, 0.0};
				cublasSafeCall(cublasZgeam(global::cublasHandle,
										   CUBLAS_OP_T,
										   CUBLAS_OP_N,
										   rows,
										   cols,
										   &alphaCu,
										   reinterpret_cast<cuDoubleComplex *>(in),
										   cols,
										   &zero,
										   reinterpret_cast<cuDoubleComplex *>(in),
										   cols,
										   reinterpret_cast<cuDoubleComplex *>(out),
										   rows));
			}
		} // namespace cuda
#endif	  // LIBRAPID_HAS_CUDA
	}	  // namespace detail

	namespace array {
		template<typename TransposeType>
		class Transpose {
		public:
			using ArrayType		 = TransposeType;
			using BaseType		 = typename std::decay_t<TransposeType>;
			using Scalar		 = typename typetraits::TypeInfo<BaseType>::Scalar;
			using ShapeType		 = typename BaseType::ShapeType;
			using Backend		 = typename typetraits::TypeInfo<BaseType>::Backend;

			static constexpr bool allowVectorisation =
			  typetraits::TypeInfo<Scalar>::allowVectorisation;
			static constexpr bool isArray  = typetraits::IsArrayContainer<BaseType>::value;
			static constexpr bool isHost   = std::is_same_v<Backend, backend::CPU>;
			static constexpr bool isOpenCL = std::is_same_v<Backend, backend::OpenCL>;
			static constexpr bool isCUDA   = std::is_same_v<Backend, backend::CUDA>;

			/// Default constructor should never be used
			Transpose() = delete;

			/// Create a Transpose object from an array/operation
			/// \param array The array to copy
			/// \param axes The transposition axes
			Transpose(TransposeType &&array, const ShapeType &axes, Scalar alpha = Scalar(1.0));

			/// Copy a Transpose object
			Transpose(const Transpose &other) = default;

			/// Move constructor
			Transpose(Transpose &&other) noexcept = default;

			/// Assign another Transpose object to this one
			/// \param other The Transpose to assign
			/// \return *this;
			auto operator=(const Transpose &other) -> Transpose & = default;

			/// Access sub-array of this Transpose object
			/// \param index Array index
			/// \return GeneralArrayView<T>
			GeneralArrayView<ArrayType> operator[](int64_t index) const;

			/// Get the shape of this Transpose object
			/// \return Shape
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ShapeType shape() const;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto size() const -> size_t;

			/// Return the number of dimensions of the Transpose object
			/// \return Number of dimensions
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE int64_t ndim() const;

			/// Access a scalar at a given index in the object. The index will be converted into
			/// a multi-dimensional index using the shape of the object, and counts in row-major
			/// order
			/// \param index Index of the scalar
			/// \return Scalar type at the given index
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto scalar(int64_t index) const;

			/// \brief Return the axes of the Transpose object
			/// \return `ShapeType` containing the axes
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const ShapeType &axes() const;

			/// \brief Return the alpha value of the Transpose object
			/// \return Alpha scaling factor
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const Scalar &alpha() const;

			/// \brief Return the untransposed array object
			/// \return Array object
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const ArrayType &array() const;
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ArrayType &array();

			template<typename ShapeType_, typename StorageType_>
			LIBRAPID_ALWAYS_INLINE void
			applyTo(ArrayContainer<ShapeType_, StorageType_> &out) const;

			/// Evaluate the Transpose object and return the result. Depending on your use case,
			/// calling this function mid-expression might result in better performance, but you
			/// should always test the available options before making a decision.
			/// \return Evaluated expression
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto eval() const;

			template<typename T, typename Char, typename Ctx>
			LIBRAPID_ALWAYS_INLINE void str(const fmt::formatter<T, Char> &format, char bracket,
											char separator, Ctx &ctx) const;

		private:
			ArrayType m_array;
			ShapeType m_inputShape;
			ShapeType m_outputShape;
			size_t m_outputSize;
			ShapeType m_axes;
			Scalar m_alpha;
		};

		template<typename T>
		Transpose<T>::Transpose(T &&array, const ShapeType &axes, Scalar alpha) :
				m_array(std::forward<T>(array)), m_inputShape(array.shape()), m_axes(axes),
				m_alpha(alpha) {
			LIBRAPID_ASSERT(m_inputShape.ndim() == m_axes.ndim(),
							"Shape and axes must have the same number of dimensions");

			m_outputShape = m_inputShape;
			for (size_t i = 0; i < m_inputShape.ndim(); i++) {
				m_outputShape[i] = m_inputShape[m_axes[i]];
			}

			m_outputSize = m_outputShape.size();
		}

		template<typename T>
		auto Transpose<T>::shape() const -> ShapeType {
			return m_outputShape;
		}

		template<typename T>
		auto Transpose<T>::size() const -> size_t {
			return m_outputSize;
		}

		template<typename T>
		auto Transpose<T>::ndim() const -> int64_t {
			return m_outputShape.ndim();
		}

		template<typename T>
		auto Transpose<T>::scalar(int64_t index) const -> auto {
			// TODO: This is a heinously inefficient way of doing this. Fix it.
			return eval().scalar(index);
		}

		template<typename T>
		auto Transpose<T>::axes() const -> const ShapeType & {
			return m_axes;
		}

		template<typename T>
		auto Transpose<T>::alpha() const -> const Scalar & {
			return m_alpha;
		}

		template<typename T>
		auto Transpose<T>::array() const -> const ArrayType & {
			return m_array;
		}

		template<typename T>
		auto Transpose<T>::array() -> ArrayType & {
			return m_array;
		}

		template<typename T>
		template<typename ShapeType_, typename StorageType_>
		void Transpose<T>::applyTo(ArrayContainer<ShapeType_, StorageType_> &out) const {
			bool inplace = ((void *)&out) == ((void *)&m_array);
			LIBRAPID_ASSERT(!inplace, "Cannot transpose inplace");
			LIBRAPID_ASSERT(out.shape() == m_outputShape, "Transpose assignment shape mismatch");

			if constexpr (isArray) {
				if constexpr (isHost) {
					auto *__restrict outPtr = out.storage().begin();
					auto *__restrict inPtr	= m_array.storage().begin();
					int64_t blockSize		= global::cacheLineSize / sizeof(Scalar);

					if (m_inputShape.ndim() == 2) {
						detail::cpu::transposeImpl(
						  outPtr, inPtr, m_inputShape[0], m_inputShape[1], m_alpha, blockSize);

					} else {
						LIBRAPID_NOT_IMPLEMENTED
					}
				}
#if defined(LIBRAPID_HAS_OPENCL)
				else if constexpr (isOpenCL) {
					cl::Buffer &outBuffer	   = out.storage().data();
					const cl::Buffer &inBuffer = m_array.storage().data();

					if (m_inputShape.ndim() == 2) {
						detail::opencl::transposeImpl<Scalar>(
						  outBuffer, inBuffer, m_inputShape[0], m_inputShape[1], m_alpha, 0);
					} else {
						LIBRAPID_NOT_IMPLEMENTED
					}
				}
#endif // LIBRAPID_HAS_OPENCL
#if defined(LIBRAPID_HAS_CUDA)
				else {
					if (m_inputShape.ndim() == 2) {
						int64_t blockSize		= global::cacheLineSize / sizeof(Scalar);
						auto *__restrict outPtr = out.storage().begin();
						auto *__restrict inPtr	= m_array.storage().begin();
						detail::cuda::transposeImpl(
						  outPtr, inPtr, m_inputShape[0], m_inputShape[1], m_alpha, blockSize);
					} else {
						LIBRAPID_NOT_IMPLEMENTED
					}
				}
#endif // LIBRAPID_HAS_CUDA
			} else {
				LIBRAPID_NOT_IMPLEMENTED
			}
		}

		template<typename T>
		auto Transpose<T>::eval() const {
			if constexpr (typetraits::TypeInfo<BaseType>::type ==
						  detail::LibRapidType::ArrayContainer) {
				using NonConstArrayType = std::remove_const_t<BaseType>;
				NonConstArrayType res(m_outputShape);
				applyTo(res);
				return res;
			} else {
				auto tmp   = m_array.eval();
				using Type = decltype(tmp);
				return Transpose<Type>(std::forward<Type>(tmp), m_axes, m_alpha).eval();
			}
		};

		template<typename TransposeType>
		template<typename T, typename Char, typename Ctx>
		void Transpose<TransposeType>::str(const fmt::formatter<T, Char> &format, char bracket,
										   char separator, Ctx &ctx) const {
			eval().str(format, bracket, separator, ctx);
		}
	}; // namespace array

	template<typename T, typename ShapeType = MatrixShape,
			 typename std::enable_if_t<typetraits::IsSizeType<ShapeType>::value, int> = 0>
	auto transpose(T &&array, const ShapeType &axes = ShapeType()) {
		// If axes is empty, transpose the array in reverse order
		ShapeType newAxes = axes;
		if (axes.size() == 0) {
			newAxes = ShapeType::zeros(array.ndim());
			for (size_t i = 0; i < array.ndim(); i++) { newAxes[i] = array.ndim() - i - 1; }
		}

		return array::Transpose<T>(std::forward<T>(array), newAxes);
	}

	namespace typetraits {
		template<typename Descriptor, typename TransposeType, typename ScalarType>
		struct HasCustomEval<detail::Function<Descriptor, detail::Multiply,
											  array::Transpose<TransposeType>, ScalarType>>
				: std::true_type {};

		template<typename Descriptor, typename ScalarType, typename TransposeType>
		struct HasCustomEval<detail::Function<Descriptor, detail::Multiply, ScalarType,
											  array::Transpose<TransposeType>>> : std::true_type {};
	}; // namespace typetraits

	namespace detail {
		// If assigning an operation of the form aT * b, where a is a matrix and b is a scalar,
		// we can replace this with Transpose(a, b) to get better performance

		// aT * b
		template<typename ShapeType, typename DestinationStorageType, typename Descriptor,
				 typename TransposeType, typename ScalarType>
		LIBRAPID_ALWAYS_INLINE void
		assign(array::ArrayContainer<ShapeType, DestinationStorageType> &destination,
			   const Function<Descriptor, detail::Multiply, array::Transpose<TransposeType>,
							  ScalarType> &function) {
			auto axes	= std::get<0>(function.args()).axes();
			auto alpha	= std::get<0>(function.args()).alpha();
			destination = array::Transpose(
			  std::get<0>(function.args()).array(), axes, alpha * std::get<1>(function.args()));
		}

		template<typename ShapeType, typename DestinationStorageType, typename Descriptor,
				 typename TransposeType, typename ScalarType>
		LIBRAPID_ALWAYS_INLINE void
		assignParallel(array::ArrayContainer<ShapeType, DestinationStorageType> &destination,
					   const Function<Descriptor, detail::Multiply, array::Transpose<TransposeType>,
									  ScalarType> &function) {
			// The assign function runs in parallel if possible by default, so just call that
			assign(destination, function);
		}

		// a * bT
		template<typename ShapeType, typename DestinationStorageType, typename ScalarType,
				 typename Descriptor, typename TransposeType>
		LIBRAPID_ALWAYS_INLINE void
		assign(array::ArrayContainer<ShapeType, DestinationStorageType> &destination,
			   const Function<Descriptor, detail::Multiply, ScalarType,
							  array::Transpose<TransposeType>> &function) {
			auto axes	= std::get<1>(function.args()).axes();
			auto alpha	= std::get<1>(function.args()).alpha();
			destination = array::Transpose(
			  std::get<1>(function.args()).array(), axes, alpha * std::get<0>(function.args()));
		}

		template<typename ShapeType, typename DestinationStorageType, typename ScalarType,
				 typename Descriptor, typename TransposeType>
		LIBRAPID_ALWAYS_INLINE void
		assignParallel(array::ArrayContainer<ShapeType, DestinationStorageType> &destination,
					   const Function<Descriptor, detail::Multiply, ScalarType,
									  array::Transpose<TransposeType>> &function) {
			assign(destination, function);
		}
	} // namespace detail
} // namespace librapid

// Support FMT printing
#ifdef FMT_API
ARRAY_TYPE_FMT_IML(typename T, librapid::array::Transpose<T>)
LIBRAPID_SIMPLE_IO_NORANGE(typename T, librapid::array::Transpose<T>)
#endif // FMT_API

#endif // LIBRAPID_ARRAY_TRANSPOSE_HPP
