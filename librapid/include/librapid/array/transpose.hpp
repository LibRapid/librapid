#ifndef LIBRAPID_ARRAY_TRANSPOSE_HPP
#define LIBRAPID_ARRAY_TRANSPOSE_HPP

namespace librapid {
	namespace typetraits {
		template<typename T>
		struct TypeInfo<array::Transpose<T>> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::Transpose;
			using Scalar							   = typename TypeInfo<std::decay_t<T>>::Scalar;
			using Device							   = typename TypeInfo<std::decay_t<T>>::Device;
			static constexpr bool allowVectorisation   = false;
		};

		LIBRAPID_DEFINE_AS_TYPE(typename T, array::Transpose<T>);
	} // namespace typetraits

	namespace kernels {
#if LIBRAPID_ARCH >= AVX2
#	define LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE 8
#	define LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE 4

		LIBRAPID_ALWAYS_INLINE void transposeFloatKernel(float *__restrict out,
														 float *__restrict in, int64_t cols) {
			__m256 r0, r1, r2, r3, r4, r5, r6, r7;
			__m256 t0, t1, t2, t3, t4, t5, t6, t7;

#	define LOAD256_IMPL(LEFT_, RIGHT_)                                                            \
		_mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(&LEFT_)), _mm_loadu_ps(&RIGHT_), 1)

			r0 = LOAD256_IMPL(in[0 * cols + 0], in[4 * cols + 0]);
			r1 = LOAD256_IMPL(in[1 * cols + 0], in[5 * cols + 0]);
			r2 = LOAD256_IMPL(in[2 * cols + 0], in[6 * cols + 0]);
			r3 = LOAD256_IMPL(in[3 * cols + 0], in[7 * cols + 0]);
			r4 = LOAD256_IMPL(in[0 * cols + 4], in[4 * cols + 4]);
			r5 = LOAD256_IMPL(in[1 * cols + 4], in[5 * cols + 4]);
			r6 = LOAD256_IMPL(in[2 * cols + 4], in[6 * cols + 4]);
			r7 = LOAD256_IMPL(in[3 * cols + 4], in[7 * cols + 4]);

#	undef LOAD256_IMPL

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

			_mm256_store_ps(&out[0 * cols], r0);
			_mm256_store_ps(&out[1 * cols], r1);
			_mm256_store_ps(&out[2 * cols], r2);
			_mm256_store_ps(&out[3 * cols], r3);
			_mm256_store_ps(&out[4 * cols], r4);
			_mm256_store_ps(&out[5 * cols], r5);
			_mm256_store_ps(&out[6 * cols], r6);
			_mm256_store_ps(&out[7 * cols], r7);
		}

		LIBRAPID_ALWAYS_INLINE void transposeDoubleKernel(double *__restrict out,
														  double *__restrict in, int64_t cols) {
			__m256d r0, r1, r2, r3;
			__m256d t0, t1, t2, t3;

#	define LOAD256_IMPL(LEFT_, RIGHT_)                                                            \
		_mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(&LEFT_)), _mm_loadu_pd(&RIGHT_), 1)

			r0 = LOAD256_IMPL(in[0 * cols + 0], in[2 * cols + 0]);
			r1 = LOAD256_IMPL(in[1 * cols + 0], in[3 * cols + 0]);
			r2 = LOAD256_IMPL(in[0 * cols + 2], in[2 * cols + 2]);
			r3 = LOAD256_IMPL(in[1 * cols + 2], in[3 * cols + 2]);

#	undef LOAD256_IMPL

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

			_mm256_store_pd(&out[0 * cols], r0);
			_mm256_store_pd(&out[1 * cols], r1);
			_mm256_store_pd(&out[2 * cols], r2);
			_mm256_store_pd(&out[3 * cols], r3);
		}
#elif LIBRAPID_ARCH >= SSE2

#	define LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE 2
#	define LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE 4

		LIBRAPID_ALWAYS_INLINE void transposeFloatKernel(float *__restrict out,
														 float *__restrict in, int64_t cols) {
			__m128 tmp3, tmp2, tmp1, tmp0;

			tmp0 = _mm_shuffle_ps(_mm_load_ps(in + 0 * cols), _mm_load_ps(in + 1 * cols), 0x44);
			tmp2 = _mm_shuffle_ps(_mm_load_ps(in + 0 * cols), _mm_load_ps(in + 1 * cols), 0xEE);
			tmp1 = _mm_shuffle_ps(_mm_load_ps(in + 2 * cols), _mm_load_ps(in + 3 * cols), 0x44);
			tmp3 = _mm_shuffle_ps(_mm_load_ps(in + 2 * cols), _mm_load_ps(in + 3 * cols), 0xEE);

			_mm_store_ps(out + 0 * cols, _mm_shuffle_ps(tmp0, tmp1, 0x88));
			_mm_store_ps(out + 1 * cols, _mm_shuffle_ps(tmp0, tmp1, 0xDD));
			_mm_store_ps(out + 2 * cols, _mm_shuffle_ps(tmp2, tmp3, 0x88));
			_mm_store_ps(out + 3 * cols, _mm_shuffle_ps(tmp2, tmp3, 0xDD));
		}

		LIBRAPID_ALWAYS_INLINE void transposeDoubleKernel(double *__restrict out,
														  double *__restrict in, int64_t cols) {
			__m128d tmp1, tmp0;

			tmp0 = _mm_shuffle_pd(_mm_load_pd(in + 0 * cols), _mm_load_pd(in + 1 * cols), 0x0);
			tmp1 = _mm_shuffle_pd(_mm_load_pd(in + 2 * cols), _mm_load_pd(in + 3 * cols), 0x0);

			_mm_store_pd(out + 0 * cols, _mm_shuffle_pd(tmp0, tmp1, 0x0));
			_mm_store_pd(out + 1 * cols, _mm_shuffle_pd(tmp0, tmp1, 0xF));
		}

#endif // LIBRAPID_MSVC
	}  // namespace kernels

	namespace detail {
		template<typename Scalar>
		LIBRAPID_ALWAYS_INLINE void transposeImpl(Scalar *__restrict out, Scalar *__restrict in,
												  int64_t rows, int64_t cols, int64_t blockSize) {
			if (rows * cols < global::multithreadThreshold) {
				for (int64_t i = 0; i < rows; i += blockSize) {
					for (int64_t j = 0; j < cols; j += blockSize) {
						for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
							for (int64_t col = j; col < j + blockSize && col < cols; ++col) {
								out[col * rows + row] = in[row * cols + col];
							}
						}
					}
				}
			} else {
#pragma omp parallel for shared(rows, cols, blockSize, in, out) default(none)                      \
  num_threads((int)global::numThreads)
				for (int64_t i = 0; i < rows; i += blockSize) {
					for (int64_t j = 0; j < cols; j += blockSize) {
						for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
							for (int64_t col = j; col < j + blockSize && col < cols; ++col) {
								out[col * rows + row] = in[row * cols + col];
							}
						}
					}
				}
			}
		}

		template<typename Scalar>
		LIBRAPID_ALWAYS_INLINE void transposeInplaceImpl(Scalar *__restrict data, int64_t rows,
														 int64_t cols, int64_t blockSize) {
			if (rows * cols < global::multithreadThreshold) {
				for (int64_t i = 0; i < rows; i += blockSize) {
					for (int64_t j = 0; j < cols; j += blockSize) {
						for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
							for (int64_t col = j; col < j + blockSize && col < cols; ++col) {
								std::swap(data[col * rows + row], data[row * cols + col]);
							}
						}
					}
				}
			} else {
#pragma omp parallel for shared(rows, cols, blockSize, data) default(none)                         \
  num_threads((int)global::numThreads)
				for (int64_t i = 0; i < rows; i += blockSize) {
					for (int64_t j = 0; j < cols; j += blockSize) {
						for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
							for (int64_t col = j; col < j + blockSize && col < cols; ++col) {
								std::swap(data[col * rows + row], data[row * cols + col]);
							}
						}
					}
				}
			}
		}

#if LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE > 0
		template<>
		LIBRAPID_ALWAYS_INLINE void transposeImpl(float *__restrict out, float *__restrict in,
												  int64_t rows, int64_t cols, int64_t) {
			constexpr int64_t blockSize = LIBRAPID_F32_TRANSPOSE_KERNEL_SIZE;
			if (rows * cols < global::multithreadThreshold) {
				for (int64_t i = 0; i < rows; i += blockSize) {
					for (int64_t j = 0; j < cols; j += blockSize) {
						if (i + blockSize <= rows && j + blockSize <= cols) {
							kernels::transposeFloatKernel(
							  &out[j * rows + i], &in[i * cols + j], rows);
						} else {
							for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
								for (int64_t col = j; col < j + blockSize && col < cols; ++col) {
									out[col * rows + row] = in[row * cols + col];
								}
							}
						}
					}
				}
			} else {
#	pragma omp parallel for shared(rows, cols, in, out) default(none)                             \
	  num_threads((int)global::numThreads)
				for (int64_t i = 0; i < rows; i += blockSize) {
					for (int64_t j = 0; j < cols; j += blockSize) {
						if (i + blockSize <= rows && j + blockSize <= cols) {
							kernels::transposeFloatKernel(
							  &out[j * rows + i], &in[i * cols + j], rows);
						} else {
							for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
								for (int64_t col = j; col < j + blockSize && col < cols; ++col) {
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
		template<>
		LIBRAPID_ALWAYS_INLINE void transposeImpl(double *__restrict out, double *__restrict in,
												  int64_t rows, int64_t cols, int64_t) {
			constexpr int64_t blockSize = LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE;
			if (rows * cols < global::multithreadThreshold) {
				for (int64_t i = 0; i < rows; i += blockSize) {
					for (int64_t j = 0; j < cols; j += blockSize) {
						if (i + blockSize <= rows && j + blockSize <= cols) {
							kernels::transposeDoubleKernel(
							  &out[j * rows + i], &in[i * cols + j], rows);
						} else {
							for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
								for (int64_t col = j; col < j + blockSize && col < cols; ++col) {
									out[col * rows + row] = in[row * cols + col];
								}
							}
						}
					}
				}
			} else {
#	pragma omp parallel for shared(rows, cols, in, out) default(none)                             \
	  num_threads((int)global::numThreads)
				for (int64_t i = 0; i < rows; i += blockSize) {
					for (int64_t j = 0; j < cols; j += blockSize) {
						if (i + blockSize <= rows && j + blockSize <= cols) {
							kernels::transposeDoubleKernel(
							  &out[j * rows + i], &in[i * cols + j], rows);
						} else {
							for (int64_t row = i; row < i + blockSize && row < rows; ++row) {
								for (int64_t col = j; col < j + blockSize && col < cols; ++col) {
									out[col * rows + row] = in[row * cols + col];
								}
							}
						}
					}
				}
			}
		}
#endif // LIBRAPID_F64_TRANSPOSE_KERNEL_SIZE > 0
	}  // namespace detail

	namespace array {
		template<typename T>
		class Transpose {
		public:
			using ArrayType		 = T;
			using BaseType		 = typename std::decay_t<T>;
			using Scalar		 = typename typetraits::TypeInfo<BaseType>::Scalar;
			using Reference		 = BaseType &;
			using ConstReference = const BaseType &;
			using ShapeType		 = typename BaseType::ShapeType;
			using Device		 = typename typetraits::TypeInfo<BaseType>::Device;

			static constexpr bool allowVectorisation =
			  typetraits::TypeInfo<Scalar>::allowVectorisation;
			static constexpr bool isArray = typetraits::IsArrayContainer<BaseType>::value;
			static constexpr bool isHost  = std::is_same_v<Device, device::CPU>;

			/// Default constructor should never be used
			Transpose() = delete;

			/// Create a Transpose object from an array/operation
			/// \param array The array to copy
			/// \param axes The transposition axes
			explicit Transpose(T &array, const ShapeType &axes);

			/// Copy a Transpose object
			Transpose(const Transpose &other) = default;

			/// Move constructor
			Transpose(Transpose &&other) = default;

			/// Assign another Transpose object to this one
			/// \param other The Transpose to assign
			/// \return *this;
			Transpose &operator=(const Transpose &other) = default;

			/// Access sub-array of this Transpose object
			/// \param index Array index
			/// \return ArrayView<T>
			ArrayView<ArrayType> operator[](int64_t index) const;

			/// Get the shape of this Transpose object
			/// \return Shape
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ShapeType shape() const;

			/// Return the number of dimensions of the Transpose object
			/// \return Number of dimensions
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE int64_t ndim() const;

			/// Access a scalar at a given index in the object. The index will be converted into
			/// a multi-dimensional index using the shape of the object, and counts in row-major
			/// order
			/// \param index Index of the scalar
			/// \return Scalar type at the given index
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto scalar(int64_t index) const;

			template<typename Container>
			LIBRAPID_ALWAYS_INLINE void applyTo(Container &out) const;

			/// Evaluate the Transpose object and return the result. Depending on your use case,
			/// calling this function mid-expression might result in better performance, but you
			/// should always test the available options before making a decision.
			/// \return Evaluated expression
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto eval() const;

			/// Return a string representation of the Transpose object, formatting each scalar with
			/// the given format string
			/// \param format Format string
			/// \return Stringified object
			LIBRAPID_NODISCARD std::string str(const std::string &format = "{}") const;

		private:
			ArrayType &m_array;
			ShapeType m_inputShape;
			ShapeType m_outputShape;
			ShapeType m_axes;
		};

		template<typename T>
		Transpose<T>::Transpose(T &array, const ShapeType &axes) :
				m_array(array), m_inputShape(array.shape()), m_axes(axes) {
			LIBRAPID_ASSERT(m_inputShape.ndim() == m_axes.ndim(),
							"Shape and axes must have the same number of dimensions");

			m_outputShape = m_inputShape;
			for (int64_t i = 0; i < m_inputShape.ndim(); i++) {
				m_outputShape[i] = m_inputShape[m_axes[i]];
			}
		}

		template<typename T>
		auto Transpose<T>::shape() const -> ShapeType {
			return m_outputShape;
		}

		template<typename T>
		template<typename Container>
		void Transpose<T>::applyTo(Container &out) const {
			bool inplace = ((void *)&out) == ((void *)&m_array);
			LIBRAPID_ASSERT(out.shape() == m_outputShape, "Transpose assignment shape mismatch");

			if constexpr (isArray && isHost && allowVectorisation) {
				auto *__restrict outPtr = out.storage().begin();
				auto *__restrict inPtr	= m_array.storage().begin();
				int64_t blockSize		= global::cacheLineSize / sizeof(Scalar);

				if (m_inputShape.ndim() == 2) {
					if (inplace) {
						detail::transposeInplaceImpl(
						  outPtr, m_inputShape[0], m_inputShape[1], blockSize);
					} else {
						detail::transposeImpl(
						  outPtr, inPtr, m_inputShape[0], m_inputShape[1], blockSize);
					}
				} else {
					LIBRAPID_NOT_IMPLEMENTED
				}
			} else {
				LIBRAPID_NOT_IMPLEMENTED
			}
		}

		template<typename T>
		auto Transpose<T>::eval() const {
			ArrayType res(m_outputShape);
			applyTo(res);
			return res;
		}

		template<typename T>
		std::string Transpose<T>::str(const std::string &format) const {
			// TODO: Optimise this for larger matrices to avoid unnecessary evaluation?
			return eval().str(format);
		}
	}; // namespace array

	template<typename T, typename ShapeType = Shape<size_t, 32>>
	auto transpose(T &&array, const ShapeType &axes = ShapeType()) {
		// If axes is empty, transpose the array in reverse order
		if (axes.ndim() == 0) {
			ShapeType tmp = ShapeType::zeros(array.ndim());
			for (int64_t i = 0; i < array.ndim(); i++) { tmp[i] = array.ndim() - i - 1; }
			return array::Transpose(array, tmp);
		}

		return array::Transpose(array, axes);
	}
} // namespace librapid

// Support FMT printing
#ifdef FMT_API
LIBRAPID_SIMPLE_IO_IMPL(typename T, librapid::array::Transpose<T>)
#endif // FMT_API

#endif // LIBRAPID_ARRAY_TRANSPOSE_HPP
