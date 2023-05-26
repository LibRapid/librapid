#ifndef LIBRAPID_ML_ACTIVATIONS
#define LIBRAPID_ML_ACTIVATIONS

namespace librapid::ml {
	// 1.	[X] Sigmoid
	// 2.	[ ] Tanh
	// 3.	[ ] ReLU
	// 4.	[ ] LeakyReLU
	// 5.	[ ] Softmax
	// 6.	[ ] Softplus
	// 7.	[ ] ELU
	// 8.	[ ] SELU
	// 9.	[ ] Swish
	// 10.	[ ] Mish
	// 11.	[ ] HardSigmoid
	// 12.	[ ] LogSigmoid
	// 13.	[ ] Softsign
	// 14.	[ ] Exponential
	// 15.	[ ] GELU
	// 16.	[ ] LogSoftmax
	// 17.	[ ] ThresholdedReLU
	// 18.	[ ] Softmin

	/// \brief Sigmoid activation function
	///
	/// A class that implements the Sigmoid activation function.
	///
	/// \f$\sigma(x) = \frac{1}{1 + e^{-x}}\f$
	///
	/// \f$\sigma'(x) = x(1 - x)\f$
	///
	class Sigmoid {
	public:
		Sigmoid() = default;

		/// Applies the Sigmoid activation function to the input array and returns the result.
		///
		/// @tparam ShapeType The type of the shape of the input array.
		/// @tparam StorageType The type of the storage of the input array.
		/// @param src The input array to apply the activation function to.
		/// @return A new array with the result of applying the Sigmoid activation function to the
		/// input array.
		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto forward(const T &src) const {
			auto ret = emptyLike(src);
			forward(ret, src);
			return ret;
		}

		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto backward(const T &src) const {
			auto ret = emptyLike(src);
			backward(ret, src);
			return ret;
		}

		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator()(const T &src) const {
			return forward(src);
		}

		/// Applies the Sigmoid activation function to the input array and stores the result in the
		/// output array.
		///
		/// @tparam ShapeType The type of the shape of the input and output arrays.
		/// @tparam StorageScalar The type of the scalar values stored in the input and output
		/// arrays.
		/// @tparam StorageAllocator The type of the allocator used to allocate memory for the input
		/// and output arrays.
		/// @param dst The output array to store the result of applying the Sigmoid activation
		/// function to the input array.
		/// @param src The input array to apply the activation function to.
		template<typename ShapeType, typename StorageScalar, typename StorageAllocator>
		LIBRAPID_ALWAYS_INLINE void
		forward(array::ArrayContainer<ShapeType, Storage<StorageScalar, StorageAllocator>> &dst,
				const array::ArrayContainer<ShapeType, Storage<StorageScalar, StorageAllocator>>
				  &src) const {
			dst = StorageScalar(1) / (StorageScalar(1) + exp(-src));
		}

		template<typename ShapeType, typename StorageScalar, typename StorageAllocator>
		LIBRAPID_ALWAYS_INLINE void
		backward(array::ArrayContainer<ShapeType, Storage<StorageScalar, StorageAllocator>> &dst,
				 const array::ArrayContainer<ShapeType, Storage<StorageScalar, StorageAllocator>>
				   &src) const {
			dst = src * (StorageScalar(1) - src);
		}

		template<typename ShapeType, typename StorageScalar, size_t... Dims>
		LIBRAPID_ALWAYS_INLINE void forward(
		  array::ArrayContainer<ShapeType, FixedStorage<StorageScalar, Dims...>> &dst,
		  const array::ArrayContainer<ShapeType, FixedStorage<StorageScalar, Dims...>> &src) const {
			dst = StorageScalar(1) / (StorageScalar(1) + exp(-src));
		}

		template<typename ShapeType, typename StorageScalar, size_t... Dims>
		LIBRAPID_ALWAYS_INLINE void backward(
		  array::ArrayContainer<ShapeType, FixedStorage<StorageScalar, Dims...>> &dst,
		  const array::ArrayContainer<ShapeType, FixedStorage<StorageScalar, Dims...>> &src) const {
			dst = src * (StorageScalar(1) - src);
		}

#if defined(LIBRAPID_HAS_OPENCL)
		template<typename ShapeType, typename StorageScalar>
		LIBRAPID_ALWAYS_INLINE void
		forward(array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &dst,
				const array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &src) const {
			opencl::runLinearKernel<StorageScalar>("sigmoidActivationForward",
												   src.shape().size(),
												   dst.storage().data(),
												   src.storage().data());
		}

		template<typename ShapeType, typename StorageScalar>
		LIBRAPID_ALWAYS_INLINE void
		backward(array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &dst,
				 const array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &src) const {
			opencl::runLinearKernel<StorageScalar>("sigmoidActivationBackward",
												   src.shape().size(),
												   dst.storage().data(),
												   src.storage().data());
		}
#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)
		template<typename ShapeType, typename StorageScalar>
		LIBRAPID_ALWAYS_INLINE void
		forward(array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &dst,
				const array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &src) const {
			cuda::runKernel<StorageScalar, StorageScalar>("activations",
														  "sigmoidActivationForward",
														  dst.shape().size(),
														  src.shape().size(),
														  dst.storage().begin().get(),
														  src.storage().begin().get());
		}

		template<typename ShapeType, typename StorageScalar>
		LIBRAPID_ALWAYS_INLINE void
		backward(array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &dst,
				 const array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &src) const {
			cuda::runKernel<StorageScalar, StorageScalar>("activations",
														  "sigmoidActivationBackward",
														  dst.shape().size(),
														  src.shape().size(),
														  dst.storage().begin().get(),
														  src.storage().begin().get());
		}
#endif // LIBRAPID_HAS_CUDA

		template<typename ShapeType, typename StorageScalar, typename StorageAllocator,
				 typename descriptor, typename Functor, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		forward(array::ArrayContainer<ShapeType, Storage<StorageScalar, StorageAllocator>> &dst,
				const detail::Function<descriptor, Functor, Args...> &src) const {
			using Type	  = detail::Function<descriptor, Functor, Args...>;
			using Backend = typename typetraits::TypeInfo<Type>::Backend;
			if constexpr (std::is_same_v<Backend, backend::CPU>) {
				dst = StorageScalar(1) / (StorageScalar(1) + exp(-src));
			} else {
				// In the case where a non-standard Backend is used, it's often faster to
				// evaluate the result and then apply a kernel directly, since it involves
				// fewer copies, which are the main cause of performance drops.
				forward(dst, src.eval());
			}
		}

		template<typename ShapeType, typename StorageScalar, typename StorageAllocator,
				 typename descriptor, typename Functor, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		backward(array::ArrayContainer<ShapeType, Storage<StorageScalar, StorageAllocator>> &dst,
				 const detail::Function<descriptor, Functor, Args...> &src) const {
			using Type	  = detail::Function<descriptor, Functor, Args...>;
			using Backend = typename typetraits::TypeInfo<Type>::Backend;
			if constexpr (std::is_same_v<Backend, backend::CPU>) {
				dst = src * (StorageScalar(1) - src);
			} else {
				backward(dst, src.eval());
			}
		}
	};
} // namespace librapid::ml

#endif // LIBRAPID_ML_ACTIVATIONS