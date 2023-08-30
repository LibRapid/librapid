#ifndef LIBRAPID_ARRAY_CUSTOM_KERNELS
#define LIBRAPID_ARRAY_CUSTOM_KERNELS

namespace librapid {
	namespace detail {
		template<typename First, typename... Rest>
		struct BackendMerger {
		private:
			static constexpr auto merger() {
				if constexpr (sizeof...(Rest) == 0) {
					using BackendFirst = typename typetraits::TypeInfo<First>::Backend;
					return BackendFirst {};
				} else {
					using BackendFirst = typename typetraits::TypeInfo<First>::Backend;
					using BackendRest  = typename BackendMerger<Rest...>::Backend;

					if constexpr (std::is_same_v<BackendFirst, BackendRest>) {
						return BackendFirst {};
					} else if constexpr (std::is_same_v<BackendFirst, backend::CUDA> ||
										 std::is_same_v<BackendRest, backend::CUDA>) {
						return backend::CUDA {};
					} else if constexpr (std::is_same_v<BackendFirst, backend::OpenCL> ||
										 std::is_same_v<BackendRest, backend::OpenCL>) {
						return backend::OpenCL {};
					} else {
						return backend::CPU {};
					}
				}
			}

		public:
			using Backend = decltype(merger());
		};
	} // namespace detail

	template<bool allowVectorisation = true, typename Functor = void>
	class Kernel {
	public:
		Kernel()				   = default;
		Kernel(const Kernel &)	   = default;
		Kernel(Kernel &&) noexcept = default;

		explicit Kernel(Functor &&functor) : m_hasFunctor(true), m_functor(functor) {}

		explicit Kernel(const std::string &openclKernel, const std::string &cudaKernel) :
				m_hasOpenCL(true), m_hasCUDA(true), m_openclKernel(openclKernel),
				m_cudaKernel(cudaKernel) {}

		explicit Kernel(Functor &&functor, const std::string &openclKernel,
						const std::string &cudaKernel) :
				m_hasFunctor(true),
				m_hasOpenCL(true), m_hasCUDA(true), m_functor(functor),
				m_openclKernel(openclKernel), m_cudaKernel(cudaKernel) {}

		~Kernel() = default;

		Kernel &operator=(const Kernel &)	  = default;
		Kernel &operator=(Kernel &&) noexcept = default;

		template<typename... Args>
		auto operator()(Args &&...args) const {
			using Backend = typename detail::BackendMerger<Args...>::Backend;

			if constexpr (std::is_same_v<Backend, backend::CPU>) {
				LIBRAPID_ASSERT(m_hasFunctor, "No functor provided for CPU kernel");
				auto func = detail::makeFunction<0, detail::descriptor::Trivial>(
				  std::forward<Functor>(m_functor), std::forward<Args>(args)...);
			}
		}

	private:
		bool m_hasFunctor = false;
		bool m_hasOpenCL  = false;
		bool m_hasCUDA	  = false;

		const Functor &m_functor;

#if defined(LIBRAPID_HAS_OPENCL)
		std::string m_openclKernel = "NONE";
#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)
		std::string m_cudaKernel = "NONE";
#endif // LIBRAPID_HAS_CUDA
	};
} // namespace librapid

#endif // LIBRAPID_ARRAY_CUSTOM_KERNELS