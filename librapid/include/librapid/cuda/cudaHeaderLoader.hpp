#ifndef LIBRAPID_CUDA_HEADER_LOADER_HPP
#define LIBRAPID_CUDA_HEADER_LOADER_HPP

namespace librapid {
	template<typename... templates, typename... Args>
	void runCudaKernel(const jitify::Program &program, const std::string &kernelName,
					   size_t elements, Args... arguments) {
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
					 .instantiate(jitify::reflection::Type<Args...>())
					 .configure(grid, block, 0, global::cudaStream)
					 .launch(arguments...));
	}
} // namespace librapid

#endif // LIBRAPID_CUDA_HEADER_LOADER_HPP