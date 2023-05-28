#ifndef LIBRAPID_CUDA_HEADER_LOADER_HPP
#define LIBRAPID_CUDA_HEADER_LOADER_HPP

#if defined(LIBRAPID_HAS_CUDA)

namespace librapid::cuda {
	/// Load a CUDA kernel from a file and return the string representation of it.
	/// \param relPath File path relative to LibRapid's "cuda/kernels" directory
	/// \return String representation of the kernel
	const std::string &loadKernel(const std::string &relPath);

	/// Run a kernel string on the GPU with the specified arguments
	/// \tparam Templates Instantiation types passed to Jitify
	/// \tparam Args Argument types passed to Jitify
	/// \param kernel Kernel string to run
	/// \param kernelName Name of the kernel
	/// \param elements Number of elements to process
	/// \param arguments Arguments to pass to the kernel
	template<typename... Templates, typename... Args>
	void runKernelString(const std::string &kernel, const std::string &kernelName, size_t elements,
						 Args... arguments) {
		jitify::Program program = global::jitCache.program(kernel);

		unsigned int threadsPerBlock, blocksPerGrid;

		// Use 1 to 512 threads per block
		if (elements < 512) {
			threadsPerBlock = static_cast<unsigned int>(elements);
			blocksPerGrid	= 1;
		} else {
			threadsPerBlock = 512;
			blocksPerGrid	= static_cast<unsigned int>(
				ceil(static_cast<double>(elements) / static_cast<double>(threadsPerBlock)));
		}

		dim3 grid(blocksPerGrid);
		dim3 block(threadsPerBlock);

#	if defined(LIBRAPID_DEBUG)
		try {
#	endif // LIBRAPID_DEBUG

			jitifyCall(program.kernel(kernelName)
						 .instantiate(jitify::reflection::Type<Templates>()...)
						 .configure(grid, block, 0, global::cudaStream)
						 .launch(arguments...));

#	if defined(LIBRAPID_DEBUG)
		} catch (const std::exception &e) {
			auto format = fmt::emphasis::bold | fmt::fg(fmt::color::red);
			fmt::print(format, "Error            : {}\n", e.what());
			fmt::print(format, "Kernel name      : {}\n", kernelName);
			fmt::print(format, "Elements         : {}\n", elements);
			fmt::print(format, "Threads per block: {}\n", threadsPerBlock);
			fmt::print(format, "Blocks per grid  : {}\n", blocksPerGrid);
			fmt::print(format, "Arguments        : {}\n", sizeof...(Args));

			// Print all arguments
			auto printer = [](auto x, auto format) {
				fmt::print(fmt::fg(fmt::color::purple), "\nArgument:\n");

				// True if x can be printed with fmt
				constexpr bool isPrintable = fmt::is_formattable<decltype(x)>::value;

				if constexpr (isPrintable) {
					fmt::print(format, "\tValue: {}\n", x);
				} else {
					fmt::print(format, "\tValue: [ CANNOT PRINT ]\n");
				}
				fmt::print(format, "\tType : {}\n", typeid(x).name());
			};

			(printer(arguments, fmt::emphasis::bold | fmt::fg(fmt::color::dark_orange)), ...);
			(printer(typeid(Templates).name(), fmt::emphasis::bold | fmt::fg(fmt::color::plum)), ...);

			// fmt::print("Information: {} {}\n", sizeof...(Templates), sizeof...(Args));

			throw;
		}
#	endif // LIBRAPID_DEBUG
	}

	/// Run a kernel from a filename and kernel name with the specified arguments
	/// \tparam Templates Instantiation types passed to Jitify
	/// \tparam Args Argument types passed to Jitify
	/// \param name Filename of the kernel
	/// \param kernelName Name of the kernel
	/// \param elements Number of elements to process
	/// \param arguments Arguments to pass to the kernel
	template<typename... Templates, typename... Args>
	void runKernel(const std::string &name, const std::string &kernelName, size_t elements,
				   Args... arguments) {
		runKernelString<Templates...>(loadKernel(name), kernelName, elements, arguments...);
	}
} // namespace librapid::cuda

#endif // LIBRAPID_HAS_CUDA

#endif // LIBRAPID_CUDA_HEADER_LOADER_HPP