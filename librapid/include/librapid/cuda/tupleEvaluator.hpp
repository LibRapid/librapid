#ifndef LIBRAPID_CUDA_TUPLE_EVALUATOR
#define LIBRAPID_CUDA_TUPLE_EVALUATOR

namespace librapid::detail {
	namespace impl {
		/// A helper class for evaluating tuples
		/// \tparam T
		template<typename T>
		struct TupleEvalHelper {
			using Type = T;
		};

		template<typename SizeType, size_t dims, typename StorageScalar>
		struct TupleEvalHelper<ArrayContainer<Shape<SizeType, dims>, CudaStorage<StorageScalar>>> {
			using Type = CudaStorage<StorageScalar>;
		};

		template<Descriptor desc, typename Functor_, typename... Args>
		struct TupleEvalHelper<Function<desc, Functor_, Args...>> {
			using Type = CudaStorage<typename Function<desc, Functor_, Args...>::Scalar>;
		};
	} // namespace impl

	/// Evaluate each element inside a tuple, allowing for
	/// \tparam First
	/// \tparam Rest
	/// \param first
	/// \param rest
	/// \return
	template<typename First, typename... Rest>
	auto cudaTupleEvaluator(First first, Rest... rest) {
		using TupleType = std::tuple<typename impl::TupleEvalHelper<First>::Type,
									 typename impl::TupleEvalHelper<Rest>::Type...>;
	}
} // namespace librapid::detail

#endif // LIBRAPID_CUDA_TUPLE_EVALUATOR