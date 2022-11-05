#ifndef LIBRAPID_CUDA_HEADER_LOADER_HPP
#define LIBRAPID_CUDA_HEADER_LOADER_HPP

namespace librapid {
	/// Load a CUDA kernel from a file and return the string representation of it.
	///
	/// \param relPath
	/// \return
	std::string loadKernel(const std::string &relPath);

	template<typename... Templates, typename... Args>
	void runKernelString(const std::string &kernel, const std::string &kernelName, size_t elements,
						 Args... arguments) {
		jitify::Program program = global::jitCache.program(kernel);

		size_t threadsPerBlock, blocksPerGrid;

		// Use 1 to 512 threads per block
		if (elements < 512) {
			threadsPerBlock = elements;
			blocksPerGrid	= 1;
		} else {
			threadsPerBlock = 512;
			blocksPerGrid	= static_cast<size_t>(
				ceil(static_cast<double>(elements) / static_cast<double>(threadsPerBlock)));
		}

		dim3 grid(blocksPerGrid);
		dim3 block(threadsPerBlock);

		jitifyCall(program.kernel(kernelName)
					 .instantiate(jitify::reflection::Type<Templates>()...)
					 .configure(grid, block, 0, global::cudaStream)
					 .launch(arguments...));
	}

	template<typename... Templates, typename... Args>
	void runKernel(const std::string &name, const std::string &kernelName, size_t elements,
				   Args... arguments) {
		runKernelString<Templates...>(loadKernel(name), kernelName, elements, arguments...);
	}
} // namespace librapid

#endif // LIBRAPID_CUDA_HEADER_LOADER_HPP