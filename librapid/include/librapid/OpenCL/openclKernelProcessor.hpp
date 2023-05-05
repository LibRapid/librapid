#ifndef LIBRAPID_OPENCL_KERNEL_PROCESSOR_HPP
#define LIBRAPID_OPENCL_KERNEL_PROCESSOR_HPP

#if defined(LIBRAPID_HAS_OPENCL)

namespace librapid::detail::impl::opencl {
	template<typename Scalar, size_t... I, typename... Args>
	void setKernelArgs(cl::Kernel &kernel, const std::tuple<Args...> &args,
					   std::index_sequence<I...>) {
		constexpr auto caster = [](auto &&x) {
			using T = std::decay_t<decltype(x)>;
			 if constexpr (std::is_same_v<T, cl::Buffer>) {
			 	return x;
			 } else if constexpr (typetraits::TypeInfo<T>::type == detail::LibRapidType::Scalar) {
			 	return static_cast<Scalar>(x);
			 } else {
			 	return x;
			 }
		};

		((kernel.setArg(I, caster(std::get<I>(args)))), ...);
	}

//	template<size_t... I, typename... Args>
//	void setKernelArgs(const std::tuple<Args...> &args, std::index_sequence<I...>) {
//		auto func = [](auto x) {
//			if constexpr (std::is_fundamental_v<std::decay_t<decltype(x)>>) {
//				fmt::print("{}\n", x);
//			} else {
//				fmt::print("{}\n", typeid(x).name());
//			}
//		};
//		((func(std::get<I>(args))), ...);
//	}

	template<typename Scalar, typename... Args>
	void runKernel(const std::string &kernelName, int64_t numElements, Args... args) {
		static_assert(sizeof(Scalar) > 2,
					  "Scalar type must be larger than 2 bytes. Please create an issue on GitHub "
					  "if you need support for smaller types.");

		std::string kernelNameFull = kernelName + "_" + typetraits::TypeInfo<Scalar>::name;
		// fmt::print("Running OpenCL kernel: {}\n", kernelNameFull);

		cl::Kernel kernel(global::openCLProgram, kernelNameFull.c_str());
		setKernelArgs<Scalar>(
		  kernel, std::make_tuple(args...), std::make_index_sequence<sizeof...(Args)>());

		cl::NDRange range(numElements);
		global::openCLQueue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange);
	}
} // namespace librapid::detail::impl::opencl

#endif // LIBRAPID_HAS_OPENCL

#endif // LIBRAPID_OPENCL_KERNEL_PROCESSOR_HPP