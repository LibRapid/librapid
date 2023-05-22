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

	class Sigmoid {
	public:
		Sigmoid() = default;

		template<typename ShapeType, typename StorageType>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		apply(const array::ArrayContainer<ShapeType, StorageType> &src) const {
			using Continer = typename std::decay_t<decltype(src)>;
			using Scalar   = typename typetraits::TypeInfo<Continer>::Scalar;
			using Backend  = typename typetraits::TypeInfo<Continer>::Backend;

			Array<Scalar, Backend> result(src.shape());
			apply(result, src);
			return result;
		}

		template<typename descriptor, typename Functor, typename... Args>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		apply(const detail::Function<descriptor, Functor, Args...> &src) const {
			using Function = detail::Function<descriptor, Functor, Args...>;
			using Scalar   = typename typetraits::TypeInfo<Function>::Scalar;
			using Backend  = typename typetraits::TypeInfo<Function>::Backend;

			Array<Scalar, Backend> result(src.shape());
			apply(result, src);
			return result;
		}

		template<typename ShapeType, typename StorageScalar, typename StorageAllocator>
		LIBRAPID_ALWAYS_INLINE void
		apply(array::ArrayContainer<ShapeType, Storage<StorageScalar, StorageAllocator>> &dst,
			  const array::ArrayContainer<ShapeType, Storage<StorageScalar, StorageAllocator>> &src)
		  const {
			dst = StorageScalar(1) / (StorageScalar(1) + exp(-src));
		}

		template<typename ShapeType, typename StorageScalar, typename StorageAllocator,
				 typename descriptor, typename Functor, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		apply(array::ArrayContainer<ShapeType, Storage<StorageScalar, StorageAllocator>> &dst,
			  const detail::Function<descriptor, Functor, Args...> &src) const {
			dst = StorageScalar(1) / (StorageScalar(1) + exp(-src));
		}

#if defined(LIBRAPID_HAS_OPENCL)
		template<typename ShapeType, typename StorageScalar, typename descriptor, typename Functor,
				 typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		apply(array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &dst,
			  const detail::Function<descriptor, Functor, Args...> &src) const {
			apply(dst, src.eval());
		}

		template<typename ShapeType, typename StorageScalar>
		LIBRAPID_ALWAYS_INLINE void
		apply(array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &dst,
			  const array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &src) const {
			opencl::runLinearKernel<StorageScalar>(
			  "sigmoidActivation", src.shape().size(), dst.storage().data(), src.storage().data());
		}
#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)
		template<typename ShapeType, typename StorageScalar, typename descriptor, typename Functor,
				 typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		apply(array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &dst,
			  const detail::Function<descriptor, Functor, Args...> &src) const {
			apply(dst, src.eval());
		}

		template<typename ShapeType, typename StorageScalar>
		LIBRAPID_ALWAYS_INLINE void
		apply(array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &dst,
			  const array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &src) const {
			cuda::runKernel<StorageScalar, StorageScalar>("activations",
														  "sigmoidActivation",
														  dst.shape().size(),
														  src.shape().size(),
														  dst.storage().begin().get(),
														  src.storage().begin().get());
		}
#endif // LIBRAPID_HAS_CUDA
	};
} // namespace librapid::ml

#endif // LIBRAPID_ML_ACTIVATIONS