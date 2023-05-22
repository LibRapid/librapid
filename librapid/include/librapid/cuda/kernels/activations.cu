template<typename Destination, typename Data>
__global__ void sigmoidActivation(size_t elements, Destination *dst, Data *src) {
	const size_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (kernelIndex < elements) { dst[kernelIndex] = 1 / (1 + exp(-src[kernelIndex])); }
}
