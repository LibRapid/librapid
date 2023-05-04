#ifndef LIBRAPID_CUDA_HEADER_LOADER_HPP
#define LIBRAPID_CUDA_HEADER_LOADER_HPP

#if defined(LIBRAPID_HAS_CUDA)

namespace librapid::detail::impl::cuda {
	/// Load a CUDA kernel from a file and return the string representation of it.
	/// \param relPath File path relative to LibRapid's "cuda/kernels" directory
	/// \return String representation of the kernel
	std::string loadKernel(const std::string &relPath);

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

		jitifyCall(program.kernel(kernelName)
					 .instantiate(jitify::reflection::Type<Templates>()...)
					 .configure(grid, block, 0, global::cudaStream)
					 .launch(arguments...));
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
} // namespace librapid

#endif // LIBRAPID_HAS_CUDA

#endif // LIBRAPID_CUDA_HEADER_LOADER_HPP