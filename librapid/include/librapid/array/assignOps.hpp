#ifndef LIBRAPID_ARRAY_ASSIGN_OPS_HPP
#define LIBRAPID_ARRAY_ASSIGN_OPS_HPP

namespace librapid {
	// All assignment operators are forward declared in "forward.hpp" so they can be used
	// elsewhere. They are defined here.

	namespace detail {
		/// Trivial array assignment operator -- assignment can be done with a single vectorised
		/// loop over contiguous data.
		/// \tparam ShapeType_ The shape type of the array container
		/// \tparam StorageScalar The scalar type of the storage object
		/// \tparam StorageAllocator The Allocator of the Storage object
		/// \tparam Functor_ The function type
		/// \tparam Args The argument types of the function
		/// \param lhs The array container to assign to
		/// \param function The function to assign
		template<typename ShapeType_, typename StorageScalar, typename StorageAllocator,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assign(array::ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>> &lhs,
			   const detail::Function<descriptor::Trivial, Functor_, Args...> &function) {
			using Function = detail::Function<descriptor::Trivial, Functor_, Args...>;
			using Scalar =
			  typename array::ArrayContainer<ShapeType_,
											 Storage<StorageScalar, StorageAllocator>>::Scalar;
			constexpr int64_t packetWidth = typetraits::TypeInfo<Scalar>::packetWidth;
			constexpr bool allowVectorisation =
			  typetraits::TypeInfo<
				detail::Function<descriptor::Trivial, Functor_, Args...>>::allowVectorisation &&
			  Function::argsAreSameType;

			const int64_t size		 = function.shape().size();
			const int64_t vectorSize = size - (size % packetWidth);

			// Ensure the function can actually be assigned to the array container
			static_assert(
			  typetraits::IsSame<Scalar, typename std::decay_t<decltype(function)>::Scalar>,
			  "Function return type must be the same as the array container's scalar type");
			LIBRAPID_ASSERT(lhs.shape() == function.shape(), "Shapes must be equal");

			if constexpr (allowVectorisation) {
				for (int64_t index = 0; index < vectorSize; index += packetWidth) {
					lhs.writePacket(index, function.packet(index));
				}

				// Assign the remaining elements
				for (int64_t index = vectorSize; index < size; ++index) {
					lhs.write(index, function.scalar(index));
				}
			} else {
				// Assign the remaining elements
				for (int64_t index = 0; index < size; ++index) {
					lhs.write(index, function.scalar(index));
				}
			}
		}

		/// Trivial assignment with fixed-size arrays
		/// \tparam ShapeType_ The shape type of the array container
		/// \tparam StorageScalar The scalar type of the storage object
		/// \tparam StorageSize The size of the storage object
		/// \tparam Functor_ The function type
		/// \tparam Args The argument types of the function
		template<typename ShapeType_, typename StorageScalar, size_t... StorageSize,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assign(array::ArrayContainer<ShapeType_, FixedStorage<StorageScalar, StorageSize...>> &lhs,
			   const detail::Function<descriptor::Trivial, Functor_, Args...> &function) {
			using Function = detail::Function<descriptor::Trivial, Functor_, Args...>;
			using Scalar =
			  typename array::ArrayContainer<ShapeType_,
											 FixedStorage<StorageScalar, StorageSize...>>::Scalar;
			constexpr int64_t packetWidth = typetraits::TypeInfo<Scalar>::packetWidth;
			constexpr int64_t elements	  = ::librapid::product<StorageSize...>();
			constexpr int64_t vectorSize  = elements - (elements % packetWidth);
			constexpr bool allowVectorisation =
			  typetraits::TypeInfo<
				detail::Function<descriptor::Trivial, Functor_, Args...>>::allowVectorisation &&
			  Function::argsAreSameType;

			// Ensure the function can actually be assigned to the array container
			static_assert(
			  typetraits::IsSame<Scalar, typename std::decay_t<decltype(function)>::Scalar>,
			  "Function return type must be the same as the array container's scalar type");
			LIBRAPID_ASSERT(lhs.shape() == function.shape(), "Shapes must be equal");

			if constexpr (allowVectorisation) {
				for (int64_t index = 0; index < vectorSize; index += packetWidth) {
					lhs.writePacket(index, function.packet(index));
				}

				// Assign the remaining elements
				for (int64_t index = vectorSize; index < elements; ++index) {
					lhs.write(index, function.scalar(index));
				}
			} else {
				// Assign the remaining elements
				for (int64_t index = 0; index < elements; ++index) {
					lhs.write(index, function.scalar(index));
				}
			}
		}

		/// Trivial assignment with parallel execution
		/// \tparam ShapeType_ The shape type of the array container
		/// \tparam StorageScalar The scalar type of the storage object
		/// \tparam StorageAllocator The Allocator of the Storage object
		/// \tparam Functor_ The function type
		/// \tparam Args The argument types of the function
		/// \param lhs The array container to assign to
		/// \param function The function to assign
		/// \see assign(array::ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>>
		/// &lhs, const detail::Function<descriptor::Trivial, Functor_, Args...> &function)
		template<typename ShapeType_, typename StorageScalar, typename StorageAllocator,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void assignParallel(
		  array::ArrayContainer<ShapeType_, Storage<StorageScalar, StorageAllocator>> &lhs,
		  const detail::Function<descriptor::Trivial, Functor_, Args...> &function) {
			using Function = detail::Function<descriptor::Trivial, Functor_, Args...>;
			using Scalar =
			  typename array::ArrayContainer<ShapeType_,
											 Storage<StorageScalar, StorageAllocator>>::Scalar;
			constexpr int64_t packetWidth = typetraits::TypeInfo<Scalar>::packetWidth;

			constexpr bool allowVectorisation =
			  typetraits::TypeInfo<
				detail::Function<descriptor::Trivial, Functor_, Args...>>::allowVectorisation &&
			  Function::argsAreSameType;

			const int64_t size		 = function.shape().size();
			const int64_t vectorSize = size - (size % packetWidth);

			// Ensure the function can actually be assigned to the array container
			static_assert(
			  typetraits::IsSame<Scalar, typename std::decay_t<decltype(function)>::Scalar>,
			  "Function return type must be the same as the array container's scalar type");
			LIBRAPID_ASSERT(lhs.shape() == function.shape(), "Shapes must be equal");

			if constexpr (allowVectorisation) {
#pragma omp parallel for shared(vectorSize, lhs, function) default(none)                           \
  num_threads(global::numThreads)
				for (int64_t index = 0; index < vectorSize; index += packetWidth) {
					lhs.writePacket(index, function.packet(index));
				}

				// Assign the remaining elements
				for (int64_t index = vectorSize; index < size; ++index) {
					lhs.write(index, function.scalar(index));
				}
			} else {
#pragma omp parallel for shared(vectorSize, lhs, function, size) default(none)                     \
  num_threads(global::numThreads)
				for (int64_t index = vectorSize; index < size; ++index) {
					lhs.write(index, function.scalar(index));
				}
			}
		}

		/// Trivial assignment with fixed-size arrays and parallel execution
		/// \tparam ShapeType_ The shape type of the array container
		/// \tparam StorageScalar The scalar type of the storage object
		/// \tparam StorageSize The size of the storage object
		/// \tparam Functor_ The function type
		/// \tparam Args The argument types of the function
		template<typename ShapeType_, typename StorageScalar, size_t... StorageSize,
				 typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void assignParallel(
		  array::ArrayContainer<ShapeType_, FixedStorage<StorageScalar, StorageSize...>> &lhs,
		  const detail::Function<descriptor::Trivial, Functor_, Args...> &function) {
			using Function = detail::Function<descriptor::Trivial, Functor_, Args...>;
			using Scalar =
			  typename array::ArrayContainer<ShapeType_,
											 FixedStorage<StorageScalar, StorageSize...>>::Scalar;
			constexpr int64_t packetWidth = typetraits::TypeInfo<Scalar>::packetWidth;

			constexpr bool allowVectorisation =
			  typetraits::TypeInfo<
				detail::Function<descriptor::Trivial, Functor_, Args...>>::allowVectorisation &&
			  Function::argsAreSameType;

			constexpr int64_t size		 = ::librapid::product<StorageSize...>();
			constexpr int64_t vectorSize = size - (size % packetWidth);

			// Ensure the function can actually be assigned to the array container
			static_assert(
			  typetraits::IsSame<Scalar, typename std::decay_t<decltype(function)>::Scalar>,
			  "Function return type must be the same as the array container's scalar type");
			LIBRAPID_ASSERT(lhs.shape() == function.shape(), "Shapes must be equal");

			if constexpr (allowVectorisation) {
#pragma omp parallel for shared(vectorSize, lhs, function) default(none)                           \
  num_threads(global::numThreads)
				for (int64_t index = 0; index < vectorSize; index += packetWidth) {
					lhs.writePacket(index, function.packet(index));
				}

				// Assign the remaining elements
				for (int64_t index = vectorSize; index < size; ++index) {
					lhs.write(index, function.scalar(index));
				}
			} else {
#pragma omp parallel for shared(vectorSize, lhs, function, size) default(none)                     \
  num_threads(global::numThreads)
				for (int64_t index = vectorSize; index < size; ++index) {
					lhs.write(index, function.scalar(index));
				}
			}
		}
	} // namespace detail

	  /*
	   * Since we cannot (reasonably) generate the kernels at runtime (ease of development,
	   * performance, etc.), operations such as (a + b) + c cannot be made into a singe kernel.
	   * Therefore, we must employ a recursive evaluator to evaluate the expression tree.
	   *
	   * Unfortunately, this is surprisingly difficult to do with the setup used by the CPU side of
	   * things.
	   *
	   * We can approach this problem as follows:
	   * 1. Create a templated function to call the kernel
	   * 2. Create a function with two specialisations
	   *    - One for an array::ArrayContainer of some kind (this is the base case)
	   *    - One for an Expression (this is the recursive case)
	   *    - The base case returns the array::ArrayContainer's storage object
	   *    - The recursive case returns the result of calling the templated function with the
	   *      Expression's left and right children
	   * 3. Call the templated function with the result of the recursive function
	   *
	   * This will be slower than a single kernel call, but it saves us from having to generate one
	   * each time, improving performance in the long run (hopefully).
	   */

#if defined(LIBRAPID_HAS_OPENCL)

	namespace opencl {
		template<typename T, typename std::enable_if_t<typetraits::TypeInfo<T>::type !=
														 ::librapid::detail::LibRapidType::Scalar,
													   int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto dataSourceExtractor(const T &obj) {
			return obj.storage().data();
		}

		template<typename T, typename std::enable_if_t<typetraits::TypeInfo<T>::type ==
														 ::librapid::detail::LibRapidType::Scalar,
													   int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto dataSourceExtractor(const T &obj) {
			return obj;
		}

		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const auto &
		openCLTupleEvaluatorImpl(const T &scalar) {
			return scalar;
		}

		template<typename ShapeType, typename StorageScalar>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const
		  array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &
		  openCLTupleEvaluatorImpl(
			const array::ArrayContainer<ShapeType, OpenCLStorage<StorageScalar>> &container) {
			return container;
		}

		template<typename descriptor, typename Functor, typename... Args>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		openCLTupleEvaluatorImpl(const detail::Function<descriptor, Functor, Args...> &function) {
			array::ArrayContainer<
			  decltype(function.shape()),
			  OpenCLStorage<typename detail::Function<descriptor, Functor, Args...>::Scalar>>
			  result(function.shape());
			assign(result, function);
			return result;
		}

		template<typename descriptor, typename Functor, typename... Args, size_t... I>
		LIBRAPID_ALWAYS_INLINE void
		openCLTupleEvaluator(std::index_sequence<I...>, const std::string &kernelBase,
							 cl::Buffer &dst,
							 const detail::Function<descriptor, Functor, Args...> &function) {
			using Scalar = typename detail::Function<descriptor, Functor, Args...>::Scalar;
			runLinearKernel<Scalar>(
			  kernelBase,
			  function.shape().size(),
			  dst,
			  dataSourceExtractor(openCLTupleEvaluatorImpl(std::get<I>(function.args())))...);
		}
	} // namespace opencl

	namespace detail {
		template<typename ShapeType_, typename StorageScalar, typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assign(array::ArrayContainer<ShapeType_, OpenCLStorage<StorageScalar>> &lhs,
			   const detail::Function<descriptor::Trivial, Functor_, Args...> &function) {
			// Unfortunately, as we are not generating the kernels at runtime, we can't use
			// temporary-free evaluation. Instead, we must recursively evaluate each sub-operation
			// until a final result is computed

			constexpr const char *filename = typetraits::TypeInfo<Functor_>::filename;
			const char *kernelBase = typetraits::TypeInfo<Functor_>::getKernelName(function.args());
			using Scalar =
			  typename array::ArrayContainer<ShapeType_, CudaStorage<StorageScalar>>::Scalar;

			const auto args			 = function.args();
			constexpr size_t argSize = std::tuple_size<decltype(args)>::value;
			::librapid::opencl::openCLTupleEvaluator(
			  std::make_index_sequence<argSize>(), kernelBase, lhs.storage().data(), function);
		}
	}  // namespace detail

#endif // LIBRAPID_HAS_CUDA

#if defined(LIBRAPID_HAS_CUDA)

	namespace cuda {
		template<typename T, typename std::enable_if_t<typetraits::TypeInfo<T>::type !=
														 ::librapid::detail::LibRapidType::Scalar,
													   int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto dataSourceExtractor(const T &obj) {
			return obj.storage().begin();
		}

		template<typename T, typename std::enable_if_t<typetraits::TypeInfo<T>::type ==
														 ::librapid::detail::LibRapidType::Scalar,
													   int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto dataSourceExtractor(const T &obj) {
			return obj;
		}

		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const auto &
		cudaTupleEvaluatorImpl(const T &scalar) {
			return scalar;
		}

		/// Helper for "evaluating" an array::ArrayContainer
		/// \tparam ShapeType The shape type of the array::ArrayContainer
		/// \tparam StorageScalar The scalar type of the array::ArrayContainer's storage object
		/// \param container The array::ArrayContainer to evaluate
		/// \return The array::ArrayContainer itself
		template<typename ShapeType, typename StorageScalar>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const
		  array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &
		  cudaTupleEvaluatorImpl(
			const array::ArrayContainer<ShapeType, CudaStorage<StorageScalar>> &container) {
			return container;
		}

		/// Helper for evaluating an expression
		/// \tparam descriptor The descriptor of the expression
		/// \tparam Functor The function type of the expression
		/// \tparam Args The argument types of the expression
		/// \param function The expression to evaluate
		/// \return The result of evaluating the expression
		template<typename descriptor, typename Functor, typename... Args>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		cudaTupleEvaluatorImpl(const detail::Function<descriptor, Functor, Args...> &function) {
			array::ArrayContainer<
			  decltype(function.shape()),
			  CudaStorage<typename detail::Function<descriptor, Functor, Args...>::Scalar>>
			  result(function.shape());
			assign(result, function);
			return result;
		}

		template<typename T>
		struct CudaVectorHelper {
			static constexpr auto tester() {
				using ScalarType = typename typetraits::TypeInfo<std::decay_t<T>>::Scalar;
				constexpr bool allowVectorisation =
				  typetraits::TypeInfo<T>::allowVectorisation;

				if constexpr (std::is_same_v<ScalarType, float> && allowVectorisation) {
					return jitify::float4 {};
				} else if constexpr (std::is_same_v<ScalarType, double> && allowVectorisation) {
					return jitify::double2 {};
				} else {
					return ScalarType {};
				}
			}

			using Scalar = decltype(tester());
		};

		/// Helper for evaluating a tuple
		/// \tparam descriptor The descriptor of the Function
		/// \tparam Functor The function type of the Function
		/// \tparam Args The argument types of the Function
		/// \tparam Pointer The pointer type of the destination
		/// \tparam I Index sequence for the tuple
		/// \param filename The filename of the kernel
		/// \param kernelName The name of the kernel
		/// \param dst The memory location to assign data to
		/// \param function The Function to evaluate
		template<typename descriptor, typename Functor, typename... Args, typename Pointer,
				 size_t... I>
		LIBRAPID_ALWAYS_INLINE void
		cudaTupleEvaluator(std::index_sequence<I...>, const std::string &filename,
						   const std::string &kernelName, Pointer *dst,
						   const detail::Function<descriptor, Functor, Args...> &function) {
			runKernel<Pointer, typename typetraits::TypeInfo<std::decay_t<Args>>::Scalar...>(
			// runKernel<Pointer, typename CudaVectorHelper<Args>::Scalar...>(
			  filename,
			  kernelName,
			  function.shape().size(),
			  function.shape().size(),
			  dst,
			  dataSourceExtractor(cudaTupleEvaluatorImpl(std::get<I>(function.args())))...);
		}
	} // namespace cuda

	namespace detail {
		/// Trivial assignment with CUDA execution
		/// \tparam ShapeType_ The shape type of the array container
		/// \tparam StorageScalar The scalar type of the storage object
		/// \tparam Functor_ The function type
		/// \tparam Args The argument types of the function
		/// \param lhs The array container to assign to
		/// \param function The function to assign
		template<typename ShapeType_, typename StorageScalar, typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE void
		assign(array::ArrayContainer<ShapeType_, CudaStorage<StorageScalar>> &lhs,
			   const detail::Function<descriptor::Trivial, Functor_, Args...> &function) {
			// Unfortunately, as we are not generating the kernels at runtime, we can't use
			// temporary-free evaluation. Instead, we must recursively evaluate each sub-operation
			// until a final result is computed

			constexpr const char *filename = typetraits::TypeInfo<Functor_>::filename;
			const char *kernelName = typetraits::TypeInfo<Functor_>::getKernelName(function.args());
			using Scalar =
			  typename array::ArrayContainer<ShapeType_, CudaStorage<StorageScalar>>::Scalar;

			const auto args			 = function.args();
			constexpr size_t argSize = std::tuple_size<decltype(args)>::value;
			::librapid::cuda::cudaTupleEvaluator(std::make_index_sequence<argSize>(),
												 filename,
												 kernelName,
												 lhs.storage().begin().get(),
												 function);
		}
	}  // namespace detail
#endif // LIBRAPID_HAS_CUDA
} // namespace librapid

#endif // LIBRAPID_ARRAY_ASSIGN_OPS_HPP